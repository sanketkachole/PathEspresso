#!/usr/bin/env python3
"""
Path-Espresso (WSI + Report) — End-to-end training script (Selector + Evidence + Reasoner)

This script is a production-ready skeleton you can plug your data into.

Expected per-patient inputs (precomputed):
- patches:        float32 [N, D]        (WSI patch embeddings)
- coords:         float32 [N, 2]        (x, y) normalized to [0, 1] if possible
- sent_emb:       float32 [M, Dt]       (sentence embeddings from report)
- claim_targets:  dict[str, int]        (Tier-1 claim targets as class indices)
- claim_sent_gt:  dict[str, List[int]]  (sentence IDs supporting each claim, from regex/IE)
- survival_time:  float32 scalar
- survival_event: int (0/1)

Key design:
- Selector picks K diverse patches conditioned on query (claim_type or survival).
- Evidence module predicts:
  - sentence evidence distribution (supervised via claim_sent_gt)
  - patch evidence distribution (weakly supervised via task consistency + sparsity)
  - claim values (supervised)
- Reasoner predicts survival risk using only evidence-selected patches/sentences + claim embeddings.

You will need to adapt:
- data IO in `PatientDataset._load_patient(...)`
- claim schema in `CLAIM_SCHEMA`
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # type: ignore


# ----------------------------
# Logging
# ----------------------------
def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Claim schema (Tier-1)
# ----------------------------
@dataclass(frozen=True)
class ClaimSpec:
    name: str
    num_classes: int  # include an "unknown" class if needed
    query_id: int     # used for conditioning (learned query embedding)


# Minimal Tier-1: adapt num_classes to your normalized labels
CLAIM_SCHEMA: List[ClaimSpec] = [
    ClaimSpec("tumor_type", 32, 0),        # bucketized / normalized tumor types
    ClaimSpec("hist_subtype", 32, 1),      # optional; can set num_classes=1 if not used
    ClaimSpec("tumor_grade", 5, 2),        # 0=unknown, 1..4=grade
    ClaimSpec("differentiation", 4, 3),    # 0=unknown, 1=well,2=moderate,3=poor
    ClaimSpec("tumor_size_bin", 4, 4),     # 0=unk, 1<=2,2=2-5,3>5
    ClaimSpec("local_invasion", 3, 5),     # 0=unk, 1=no, 2=yes
    ClaimSpec("vascular_invasion", 3, 6),  # 0=unk, 1=no, 2=yes
    ClaimSpec("margins", 3, 7),            # 0=unk, 1=neg,2=pos
    ClaimSpec("node_involvement", 3, 8),   # 0=unk/not assessed,1=absent,2=present
    ClaimSpec("pT", 12, 9),                # bucket pT: unk, Tis, T0..T4a.. etc
    ClaimSpec("pN", 6, 10),                # unk, N0,N1,N2,N3,Nx
    ClaimSpec("stage_group", 6, 11),       # unk, I,II,III,IV, other
]

# Special query for survival selection
SURVIVAL_QUERY_ID = 1000


# ----------------------------
# Utilities
# ----------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """
    logits: [..., L]
    mask:   [..., L]  bool or 0/1
    """
    mask = mask.to(dtype=torch.bool)
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=dim)


def topk_diverse(
    emb: Tensor,
    scores: Tensor,
    k: int,
    pool: int,
    min_cosine_distance: float = 0.05,
) -> Tensor:
    """
    Greedy diverse selection:
    - take top `pool` by score
    - greedily add items that are sufficiently different in cosine space
    Returns indices into the original set.
    """
    if emb.ndim != 2 or scores.ndim != 1:
        raise ValueError("emb must be [N, H], scores must be [N].")
    n = emb.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=emb.device)

    pool = min(pool, n)
    top_pool = torch.topk(scores, k=pool, largest=True).indices  # [pool]
    pool_emb = F.normalize(emb[top_pool], dim=-1)                # [pool, H]

    selected: List[int] = []
    for i in range(pool):
        if len(selected) >= k:
            break
        if not selected:
            selected.append(i)
            continue
        cand = pool_emb[i]
        sel = pool_emb[torch.tensor(selected, device=emb.device)]
        cos_sim = (sel @ cand).max().item()
        if (1.0 - cos_sim) >= min_cosine_distance:
            selected.append(i)

    # If diversity is too strict, pad with remaining top items
    if len(selected) < k:
        for i in range(pool):
            if i not in selected:
                selected.append(i)
            if len(selected) >= k:
                break

    selected_pool_idx = torch.tensor(selected[:k], device=emb.device, dtype=torch.long)
    return top_pool[selected_pool_idx]  # indices into original N


# ----------------------------
# Data
# ----------------------------
class PatientDataset(Dataset[Dict[str, Any]]):
    """
    Dataset reads one patient per item.

    Data format options:
    - One .npz per patient in root/{split}/<patient_id>.npz
      Keys: patches, coords, sent_emb, claim_targets_json, claim_sent_gt_json, surv_time, surv_event
    - Or a single .pt/.pth file containing a list of dicts (set --format ptlist).
    """

    def __init__(
        self,
        root: Path,
        split: str,
        fmt: str,
        max_patches: int,
        max_sents: int,
    ) -> None:
        self.root = root
        self.split = split
        self.fmt = fmt
        self.max_patches = max_patches
        self.max_sents = max_sents

        if fmt == "npz":
            self.items = sorted((root / split).glob("*.npz"))
        elif fmt == "ptlist":
            p = root / f"{split}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
            self.items = torch.load(p)
        else:
            raise ValueError("fmt must be one of: npz, ptlist")

        if not self.items:
            raise RuntimeError(f"No data found for split={split} fmt={fmt} at {root}")

    def __len__(self) -> int:
        return len(self.items)

    def _truncate(self, arr: np.ndarray, max_len: int) -> np.ndarray:
        if arr.shape[0] <= max_len:
            return arr
        return arr[:max_len]

    def _load_npz(self, path: Path) -> Dict[str, Any]:
        data = np.load(path, allow_pickle=True)
        patches = self._truncate(data["patches"].astype(np.float32), self.max_patches)
        coords = self._truncate(data["coords"].astype(np.float32), self.max_patches)
        sent_emb = self._truncate(data["sent_emb"].astype(np.float32), self.max_sents)

        claim_targets = json.loads(str(data["claim_targets_json"]))
        claim_sent_gt = json.loads(str(data["claim_sent_gt_json"]))

        surv_time = float(data["surv_time"])
        surv_event = int(data["surv_event"])

        patient_id = path.stem
        return {
            "patient_id": patient_id,
            "patches": patches,
            "coords": coords,
            "sent_emb": sent_emb,
            "claim_targets": claim_targets,
            "claim_sent_gt": claim_sent_gt,
            "surv_time": surv_time,
            "surv_event": surv_event,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.fmt == "npz":
            return self._load_npz(Path(self.items[idx]))
        # ptlist
        item = self.items[idx]
        # You may want to truncate here too.
        return item


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Variable lengths → pad
    patient_ids = [b["patient_id"] for b in batch]

    patches_list = [torch.from_numpy(b["patches"]) for b in batch]
    coords_list = [torch.from_numpy(b["coords"]) for b in batch]
    sent_list = [torch.from_numpy(b["sent_emb"]) for b in batch]

    bsz = len(batch)
    max_n = max(p.shape[0] for p in patches_list)
    max_m = max(s.shape[0] for s in sent_list)
    d = patches_list[0].shape[1]
    dt = sent_list[0].shape[1]

    patches = torch.zeros((bsz, max_n, d), dtype=torch.float32)
    coords = torch.zeros((bsz, max_n, 2), dtype=torch.float32)
    patch_mask = torch.zeros((bsz, max_n), dtype=torch.bool)

    sent_emb = torch.zeros((bsz, max_m, dt), dtype=torch.float32)
    sent_mask = torch.zeros((bsz, max_m), dtype=torch.bool)

    for i in range(bsz):
        n = patches_list[i].shape[0]
        m = sent_list[i].shape[0]
        patches[i, :n] = patches_list[i]
        coords[i, :n] = coords_list[i]
        patch_mask[i, :n] = True
        sent_emb[i, :m] = sent_list[i]
        sent_mask[i, :m] = True

    # Claims + sentence GT are dicts (ragged) kept as Python objects
    claim_targets = [b["claim_targets"] for b in batch]
    claim_sent_gt = [b["claim_sent_gt"] for b in batch]

    surv_time = torch.tensor([float(b["surv_time"]) for b in batch], dtype=torch.float32)
    surv_event = torch.tensor([int(b["surv_event"]) for b in batch], dtype=torch.float32)

    return {
        "patient_id": patient_ids,
        "patches": patches,
        "coords": coords,
        "patch_mask": patch_mask,
        "sent_emb": sent_emb,
        "sent_mask": sent_mask,
        "claim_targets": claim_targets,
        "claim_sent_gt": claim_sent_gt,
        "surv_time": surv_time,
        "surv_event": surv_event,
    }


# ----------------------------
# Models
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PatchSelector(nn.Module):
    """
    Scores patches conditioned on query embedding and selects K diverse patches.
    """

    def __init__(
        self,
        patch_dim: int,
        hidden: int,
        proj_dim: int,
        num_queries: int,
        coord_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_emb = nn.Embedding(num_queries, proj_dim)
        self.patch_proj = MLP(patch_dim, hidden, proj_dim, dropout)
        self.coord_proj = MLP(2, hidden // 2, coord_dim, dropout)
        self.score = MLP(proj_dim + coord_dim + proj_dim, hidden, 1, dropout)

    def forward(
        self,
        patches: Tensor,        # [B, N, D]
        coords: Tensor,         # [B, N, 2]
        patch_mask: Tensor,     # [B, N]
        query_ids: Tensor,      # [B]
        k: int,
        pool: int,
        min_cos_dist: float,
    ) -> Dict[str, Tensor]:
        bsz, n, _ = patches.shape
        q = self.query_emb(query_ids)  # [B, H]
        patch_h = self.patch_proj(patches)  # [B, N, H]
        coord_h = self.coord_proj(coords)   # [B, N, C]
        q_exp = q.unsqueeze(1).expand(bsz, n, q.shape[-1])
        feat = torch.cat([patch_h, coord_h, q_exp], dim=-1)  # [B, N, H+C+H]
        logits = self.score(feat).squeeze(-1)                # [B, N]
        logits = logits.masked_fill(~patch_mask, float("-inf"))

        # Select per sample
        sel_indices = []
        sel_scores = []
        for b in range(bsz):
            valid_idx = patch_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                sel_indices.append(torch.zeros((0,), dtype=torch.long, device=patches.device))
                sel_scores.append(torch.zeros((0,), dtype=torch.float32, device=patches.device))
                continue
            emb_b = patch_h[b, valid_idx]      # [Nv, H]
            scores_b = logits[b, valid_idx]    # [Nv]
            idx_in_valid = topk_diverse(
                emb=emb_b,
                scores=scores_b,
                k=min(k, valid_idx.numel()),
                pool=min(pool, valid_idx.numel()),
                min_cosine_distance=min_cos_dist,
            )
            chosen = valid_idx[idx_in_valid]  # back to original N space
            sel_indices.append(chosen)
            sel_scores.append(logits[b, chosen])

        # Pad selection indices to K
        max_k = max(x.numel() for x in sel_indices)
        max_k = max(max_k, 1)
        out_idx = torch.full((bsz, max_k), 0, dtype=torch.long, device=patches.device)
        out_mask = torch.zeros((bsz, max_k), dtype=torch.bool, device=patches.device)
        out_score = torch.zeros((bsz, max_k), dtype=torch.float32, device=patches.device)
        for b in range(bsz):
            kk = sel_indices[b].numel()
            if kk == 0:
                continue
            out_idx[b, :kk] = sel_indices[b]
            out_mask[b, :kk] = True
            out_score[b, :kk] = sel_scores[b]

        return {
            "sel_idx": out_idx,       # [B, Ksel]
            "sel_mask": out_mask,     # [B, Ksel]
            "sel_logits": out_score,  # [B, Ksel]
        }


class EvidenceModule(nn.Module):
    """
    Takes selected patches + sentence embeddings and outputs:
    - sentence evidence distribution (supervised)
    - patch evidence distribution (weak supervision + sparsity)
    - claim predictions (supervised)
    """

    def __init__(
        self,
        patch_dim: int,
        sent_dim: int,
        hidden: int,
        num_queries: int,
        n_layers_patch: int = 2,
        n_layers_sent: int = 1,
        n_layers_cross: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_emb = nn.Embedding(num_queries, hidden)

        self.patch_in = nn.Linear(patch_dim, hidden)
        self.sent_in = nn.Linear(sent_dim, hidden)

        encoder_layer_patch = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4, dropout=dropout, batch_first=True
        )
        self.patch_enc = nn.TransformerEncoder(encoder_layer_patch, num_layers=n_layers_patch)

        encoder_layer_sent = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4, dropout=dropout, batch_first=True
        )
        self.sent_enc = nn.TransformerEncoder(encoder_layer_sent, num_layers=n_layers_sent)

        self.cross_pt = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden, num_heads=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers_cross)
        ])
        self.cross_tp = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden, num_heads=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers_cross)
        ])
        self.norm_p = nn.LayerNorm(hidden)
        self.norm_t = nn.LayerNorm(hidden)

        self.patch_evidence_head = nn.Linear(hidden, 1)
        self.sent_evidence_head = nn.Linear(hidden, 1)

        # Per-claim classifier heads
        self.claim_heads = nn.ModuleDict({
            c.name: nn.Linear(hidden, c.num_classes) for c in CLAIM_SCHEMA
        })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sel_patches: Tensor,     # [B, K, D]
        sel_mask: Tensor,        # [B, K]
        sent_emb: Tensor,        # [B, M, Dt]
        sent_mask: Tensor,       # [B, M]
        query_ids: Tensor,       # [B]
    ) -> Dict[str, Any]:
        bsz = sel_patches.shape[0]
        q = self.query_emb(query_ids).unsqueeze(1)  # [B,1,H]

        p = self.patch_in(sel_patches)  # [B,K,H]
        t = self.sent_in(sent_emb)      # [B,M,H]

        # Encode within modality
        p = self.patch_enc(p, src_key_padding_mask=~sel_mask)
        t = self.sent_enc(t, src_key_padding_mask=~sent_mask)

        # Cross attention stacks
        p = self.norm_p(p + q.expand(bsz, p.shape[1], q.shape[-1]) * 0.0)  # keep shape; q used later if desired
        t = self.norm_t(t)

        for attn_pt, attn_tp in zip(self.cross_pt, self.cross_tp):
            # Patch attends to sentences
            p2, _ = attn_pt(
                query=p,
                key=t,
                value=t,
                key_padding_mask=~sent_mask,
                need_weights=False,
            )
            p = self.norm_p(p + self.dropout(p2))

            # Sentences attend to patches
            t2, _ = attn_tp(
                query=t,
                key=p,
                value=p,
                key_padding_mask=~sel_mask,
                need_weights=False,
            )
            t = self.norm_t(t + self.dropout(t2))

        patch_logits = self.patch_evidence_head(p).squeeze(-1)  # [B,K]
        sent_logits = self.sent_evidence_head(t).squeeze(-1)    # [B,M]

        patch_probs = masked_softmax(patch_logits, sel_mask, dim=-1)
        sent_probs = masked_softmax(sent_logits, sent_mask, dim=-1)

        # Pooled representation for claim prediction (use both)
        p_pool = (patch_probs.unsqueeze(-1) * p).sum(dim=1)  # [B,H]
        t_pool = (sent_probs.unsqueeze(-1) * t).sum(dim=1)   # [B,H]
        fused = torch.cat([p_pool, t_pool], dim=-1)          # [B,2H]
        fused = nn.Linear(fused.shape[-1], p_pool.shape[-1]).to(fused.device)(fused)  # lightweight fusion
        fused = F.gelu(fused)

        claim_logits: Dict[str, Tensor] = {}
        for c in CLAIM_SCHEMA:
            claim_logits[c.name] = self.claim_heads[c.name](fused)  # [B, num_classes]

        return {
            "patch_logits": patch_logits,
            "patch_probs": patch_probs,
            "sent_logits": sent_logits,
            "sent_probs": sent_probs,
            "claim_logits": claim_logits,
            "fused": fused,
        }


class Reasoner(nn.Module):
    """
    Final predictor uses evidence-selected patches/sentences + predicted claims (as embeddings)
    to output survival risk.
    """

    def __init__(
        self,
        hidden: int,
        claim_embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Embed predicted claim values (class ids)
        self.claim_emb_tables = nn.ModuleDict({
            c.name: nn.Embedding(c.num_classes, claim_embed_dim) for c in CLAIM_SCHEMA
        })

        self.patch_pool = nn.Linear(hidden, hidden)
        self.sent_pool = nn.Linear(hidden, hidden)
        self.claim_pool = nn.Linear(claim_embed_dim, hidden)

        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(hidden, 1)  # Cox risk score

    def forward(
        self,
        patch_repr: Tensor,  # [B,H] pooled
        sent_repr: Tensor,   # [B,H] pooled
        pred_claim_ids: Dict[str, Tensor],  # each [B]
    ) -> Tensor:
        claim_vecs = []
        for c in CLAIM_SCHEMA:
            ids = pred_claim_ids[c.name]
            claim_vecs.append(self.claim_emb_tables[c.name](ids))  # [B,E]
        claim_mat = torch.stack(claim_vecs, dim=1)  # [B,C,E]
        claim_repr = claim_mat.mean(dim=1)          # [B,E]

        p = F.gelu(self.patch_pool(patch_repr))
        t = F.gelu(self.sent_pool(sent_repr))
        cvec = F.gelu(self.claim_pool(claim_repr))

        fused = self.fuse(torch.cat([p, t, cvec], dim=-1))
        risk = self.risk_head(fused).squeeze(-1)  # [B]
        return risk


class PathEspresso(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        sent_dim: int,
        hidden: int,
        selector_proj: int,
        max_query_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_queries = max_query_id + 1
        self.selector = PatchSelector(
            patch_dim=patch_dim,
            hidden=hidden,
            proj_dim=selector_proj,
            num_queries=self.num_queries,
            dropout=dropout,
        )
        self.evidence = EvidenceModule(
            patch_dim=patch_dim,
            sent_dim=sent_dim,
            hidden=hidden,
            num_queries=self.num_queries,
            dropout=dropout,
        )
        self.reasoner = Reasoner(hidden=hidden, claim_embed_dim=hidden // 2, dropout=dropout)

    def forward(
        self,
        patches: Tensor,
        coords: Tensor,
        patch_mask: Tensor,
        sent_emb: Tensor,
        sent_mask: Tensor,
        query_ids: Tensor,
        k: int,
        pool: int,
        min_cos_dist: float,
        evidence_top_patches: int,
        evidence_top_sents: int,
    ) -> Dict[str, Any]:
        sel = self.selector(
            patches=patches,
            coords=coords,
            patch_mask=patch_mask,
            query_ids=query_ids,
            k=k,
            pool=pool,
            min_cos_dist=min_cos_dist,
        )
        sel_idx = sel["sel_idx"]     # [B,Ksel]
        sel_mask = sel["sel_mask"]   # [B,Ksel]

        # Gather selected patches
        bsz, ksel = sel_idx.shape
        idx_exp = sel_idx.unsqueeze(-1).expand(bsz, ksel, patches.shape[-1])
        sel_patches = torch.gather(patches, dim=1, index=idx_exp)  # [B,Ksel,D]

        ev = self.evidence(
            sel_patches=sel_patches,
            sel_mask=sel_mask,
            sent_emb=sent_emb,
            sent_mask=sent_mask,
            query_ids=query_ids,
        )

        # Choose evidence items (hard selection for faithfulness)
        patch_probs = ev["patch_probs"]  # [B,Ksel]
        sent_probs = ev["sent_probs"]    # [B,M]

        patch_top = torch.topk(
            patch_probs.masked_fill(~sel_mask, 0.0),
            k=min(evidence_top_patches, ksel),
            dim=-1,
        ).indices  # [B,ep]
        sent_top = torch.topk(
            sent_probs.masked_fill(~sent_mask, 0.0),
            k=min(evidence_top_sents, sent_emb.shape[1]),
            dim=-1,
        ).indices  # [B,es]

        # Pooled representations using *only* top evidence
        # Patches
        patch_ev = torch.gather(sel_patches, 1, patch_top.unsqueeze(-1).expand(-1, -1, sel_patches.shape[-1]))
        patch_repr = patch_ev.mean(dim=1)  # [B,D] but Evidence uses hidden internally; keep simple: use fused
        # Sentences
        sent_ev = torch.gather(sent_emb, 1, sent_top.unsqueeze(-1).expand(-1, -1, sent_emb.shape[-1]))
        sent_repr = sent_ev.mean(dim=1)  # [B,Dt]

        # Map to hidden space for reasoner by reusing evidence fused vector (more stable)
        fused = ev["fused"]  # [B,H]
        patch_repr_h = fused
        sent_repr_h = fused

        # Predicted claim ids
        pred_claim_ids: Dict[str, Tensor] = {}
        for c in CLAIM_SCHEMA:
            pred_claim_ids[c.name] = torch.argmax(ev["claim_logits"][c.name], dim=-1)

        risk = self.reasoner(
            patch_repr=patch_repr_h,
            sent_repr=sent_repr_h,
            pred_claim_ids=pred_claim_ids,
        )

        return {
            "sel": sel,
            "ev": ev,
            "risk": risk,
            "pred_claim_ids": pred_claim_ids,
            "evidence_patch_top": patch_top,
            "evidence_sent_top": sent_top,
        }


# ----------------------------
# Losses
# ----------------------------
def cox_ph_loss(risk: Tensor, time_days: Tensor, event: Tensor) -> Tensor:
    """
    Cox partial likelihood loss.
    risk:  [B] higher = more risk
    time:  [B]
    event: [B] {0,1}
    """
    # Sort by time descending
    order = torch.argsort(time_days, descending=True)
    r = risk[order]
    e = event[order]
    # log cumulative hazard
    log_cumsum = torch.logcumsumexp(r, dim=0)
    # negative partial log-likelihood
    loss = -torch.sum((r - log_cumsum) * e) / (e.sum() + 1e-8)
    return loss


def sentence_evidence_loss(
    sent_logits: Tensor,          # [B,M]
    sent_mask: Tensor,            # [B,M]
    claim_sent_gt: List[Dict[str, List[int]]],
    claim_name: str,
) -> Tensor:
    """
    Supervise sentence evidence for a claim using multi-positive NLL over the sentence distribution.
    If no GT for a sample, skip it.
    """
    bsz, m = sent_logits.shape
    loss_vals: List[Tensor] = []
    log_probs = torch.log(masked_softmax(sent_logits, sent_mask, dim=-1) + 1e-12)  # [B,M]

    for b in range(bsz):
        gt = claim_sent_gt[b].get(claim_name, [])
        gt = [i for i in gt if 0 <= i < m and bool(sent_mask[b, i].item())]
        if not gt:
            continue
        # multi-positive: -log(sum p(gt))
        lp = torch.logsumexp(log_probs[b, torch.tensor(gt, device=sent_logits.device)], dim=0)
        loss_vals.append(-lp)

    if not loss_vals:
        return torch.zeros((), device=sent_logits.device)
    return torch.stack(loss_vals).mean()


def claim_loss(
    claim_logits: Dict[str, Tensor],
    claim_targets: List[Dict[str, int]],
    device: torch.device,
) -> Tensor:
    losses: List[Tensor] = []
    for c in CLAIM_SCHEMA:
        logits = claim_logits[c.name]  # [B,C]
        tgt = torch.tensor(
            [int(t.get(c.name, 0)) for t in claim_targets],
            dtype=torch.long,
            device=device,
        )
        losses.append(F.cross_entropy(logits, tgt))
    return torch.stack(losses).mean()


def sparsity_loss(probs: Tensor, mask: Tensor) -> Tensor:
    """
    Encourage peaky evidence: sum p^2 (maximized) → minimize negative.
    """
    probs = probs * mask.to(probs.dtype)
    return -torch.mean(torch.sum(probs**2, dim=-1))


# ----------------------------
# Train / Eval
# ----------------------------
@dataclass
class TrainConfig:
    data_root: Path
    fmt: str
    output_dir: Path
    seed: int
    device: str

    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    wd: float
    grad_clip: float
    fp16: bool

    max_patches: int
    max_sents: int
    selector_k: int
    selector_pool: int
    selector_min_cos_dist: float
    evidence_top_patches: int
    evidence_top_sents: int

    hidden: int
    selector_proj: int
    dropout: float

    lambda_surv: float
    lambda_claim: float
    lambda_sent_ev: float
    lambda_sparse_patch: float


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_root=Path(args.data_root),
        fmt=args.format,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        grad_clip=args.grad_clip,
        fp16=args.fp16,
        max_patches=args.max_patches,
        max_sents=args.max_sents,
        selector_k=args.selector_k,
        selector_pool=args.selector_pool,
        selector_min_cos_dist=args.selector_min_cos_dist,
        evidence_top_patches=args.evidence_top_patches,
        evidence_top_sents=args.evidence_top_sents,
        hidden=args.hidden,
        selector_proj=args.selector_proj,
        dropout=args.dropout,
        lambda_surv=args.lambda_surv,
        lambda_claim=args.lambda_claim,
        lambda_sent_ev=args.lambda_sent_ev,
        lambda_sparse_patch=args.lambda_sparse_patch,
    )


def compute_max_query_id() -> int:
    max_claim_q = max(c.query_id for c in CLAIM_SCHEMA) if CLAIM_SCHEMA else 0
    return max(max_claim_q, SURVIVAL_QUERY_ID)


@torch.no_grad()
def eval_one_epoch(
    model: PathEspresso,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    total = 0
    surv_loss_sum = 0.0
    claim_loss_sum = 0.0
    sent_loss_sum = 0.0
    sparse_sum = 0.0

    for batch in loader:
        patches = batch["patches"].to(device)
        coords = batch["coords"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        sent_emb = batch["sent_emb"].to(device)
        sent_mask = batch["sent_mask"].to(device)
        claim_targets = batch["claim_targets"]
        claim_sent_gt = batch["claim_sent_gt"]
        surv_time = batch["surv_time"].to(device)
        surv_event = batch["surv_event"].to(device)

        # Use survival query at eval
        query_ids = torch.full((patches.shape[0],), SURVIVAL_QUERY_ID, dtype=torch.long, device=device)

        out = model(
            patches=patches,
            coords=coords,
            patch_mask=patch_mask,
            sent_emb=sent_emb,
            sent_mask=sent_mask,
            query_ids=query_ids,
            k=cfg.selector_k,
            pool=cfg.selector_pool,
            min_cos_dist=cfg.selector_min_cos_dist,
            evidence_top_patches=cfg.evidence_top_patches,
            evidence_top_sents=cfg.evidence_top_sents,
        )

        risk = out["risk"]
        ev = out["ev"]

        l_surv = cox_ph_loss(risk, surv_time, surv_event)
        l_claim = claim_loss(ev["claim_logits"], claim_targets, device=device)

        # Sentence evidence supervision: average over claims with GT
        sent_losses = []
        for c in CLAIM_SCHEMA:
            sent_losses.append(sentence_evidence_loss(
                sent_logits=ev["sent_logits"],
                sent_mask=sent_mask,
                claim_sent_gt=claim_sent_gt,
                claim_name=c.name,
            ))
        l_sent = torch.stack(sent_losses).mean()

        l_sparse = sparsity_loss(ev["patch_probs"], out["sel"]["sel_mask"])

        total += patches.shape[0]
        surv_loss_sum += float(l_surv.item()) * patches.shape[0]
        claim_loss_sum += float(l_claim.item()) * patches.shape[0]
        sent_loss_sum += float(l_sent.item()) * patches.shape[0]
        sparse_sum += float(l_sparse.item()) * patches.shape[0]

    if total == 0:
        return {"surv_loss": 0.0, "claim_loss": 0.0, "sent_ev_loss": 0.0, "sparsity_loss": 0.0}
    return {
        "surv_loss": surv_loss_sum / total,
        "claim_loss": claim_loss_sum / total,
        "sent_ev_loss": sent_loss_sum / total,
        "sparsity_loss": sparse_sum / total,
    }


def train_one_epoch(
    model: PathEspresso,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    use_wandb: bool,
) -> Dict[str, float]:
    model.train()
    total = 0
    loss_sum = 0.0

    surv_loss_sum = 0.0
    claim_loss_sum = 0.0
    sent_loss_sum = 0.0
    sparse_sum = 0.0

    start = time.time()

    for step, batch in enumerate(loader):
        patches = batch["patches"].to(device)
        coords = batch["coords"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        sent_emb = batch["sent_emb"].to(device)
        sent_mask = batch["sent_mask"].to(device)
        claim_targets = batch["claim_targets"]
        claim_sent_gt = batch["claim_sent_gt"]
        surv_time = batch["surv_time"].to(device)
        surv_event = batch["surv_event"].to(device)

        # Training: you can alternate queries:
        # - survival query for selector/evidence
        # - or per-claim query_ids for multi-task supervision
        # Minimal: use survival query for all
        query_ids = torch.full((patches.shape[0],), SURVIVAL_QUERY_ID, dtype=torch.long, device=device)

        optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            out = model(
                patches=patches,
                coords=coords,
                patch_mask=patch_mask,
                sent_emb=sent_emb,
                sent_mask=sent_mask,
                query_ids=query_ids,
                k=cfg.selector_k,
                pool=cfg.selector_pool,
                min_cos_dist=cfg.selector_min_cos_dist,
                evidence_top_patches=cfg.evidence_top_patches,
                evidence_top_sents=cfg.evidence_top_sents,
            )

            risk = out["risk"]
            ev = out["ev"]

            l_surv = cox_ph_loss(risk, surv_time, surv_event)
            l_claim = claim_loss(ev["claim_logits"], claim_targets, device=device)

            sent_losses = []
            for c in CLAIM_SCHEMA:
                sent_losses.append(sentence_evidence_loss(
                    sent_logits=ev["sent_logits"],
                    sent_mask=sent_mask,
                    claim_sent_gt=claim_sent_gt,
                    claim_name=c.name,
                ))
            l_sent = torch.stack(sent_losses).mean()

            l_sparse = sparsity_loss(ev["patch_probs"], out["sel"]["sel_mask"])

            loss = (
                cfg.lambda_surv * l_surv
                + cfg.lambda_claim * l_claim
                + cfg.lambda_sent_ev * l_sent
                + cfg.lambda_sparse_patch * l_sparse
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        bsz = patches.shape[0]
        total += bsz
        loss_sum += float(loss.item()) * bsz
        surv_loss_sum += float(l_surv.item()) * bsz
        claim_loss_sum += float(l_claim.item()) * bsz
        sent_loss_sum += float(l_sent.item()) * bsz
        sparse_sum += float(l_sparse.item()) * bsz

        if use_wandb and (step % 20 == 0):
            wandb.log({
                "train/loss": float(loss.item()),
                "train/surv_loss": float(l_surv.item()),
                "train/claim_loss": float(l_claim.item()),
                "train/sent_ev_loss": float(l_sent.item()),
                "train/sparsity_loss": float(l_sparse.item()),
                "epoch": epoch,
                "step": epoch * len(loader) + step,
            })

    elapsed = time.time() - start
    metrics = {
        "loss": loss_sum / max(total, 1),
        "surv_loss": surv_loss_sum / max(total, 1),
        "claim_loss": claim_loss_sum / max(total, 1),
        "sent_ev_loss": sent_loss_sum / max(total, 1),
        "sparsity_loss": sparse_sum / max(total, 1),
        "sec_per_epoch": elapsed,
    }
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "cfg": cfg.__dict__,
        "claim_schema": [c.__dict__ for c in CLAIM_SCHEMA],
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-Espresso end-to-end training")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--format", type=str, default="npz", choices=["npz", "ptlist"])
    parser.add_argument("--output_dir", type=str, default="runs/path_espresso")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--max_patches", type=int, default=8000)
    parser.add_argument("--max_sents", type=int, default=256)

    parser.add_argument("--selector_k", type=int, default=64)
    parser.add_argument("--selector_pool", type=int, default=512)
    parser.add_argument("--selector_min_cos_dist", type=float, default=0.05)

    parser.add_argument("--evidence_top_patches", type=int, default=8)
    parser.add_argument("--evidence_top_sents", type=int, default=3)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--selector_proj", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda_surv", type=float, default=1.0)
    parser.add_argument("--lambda_claim", type=float, default=1.0)
    parser.add_argument("--lambda_sent_ev", type=float, default=1.0)
    parser.add_argument("--lambda_sparse_patch", type=float, default=0.05)

    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")

    args = parser.parse_args()
    cfg = build_config(args)

    setup_logging("INFO")
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _ensure_dir(cfg.output_dir)

    # Data
    train_ds = PatientDataset(cfg.data_root, "train", cfg.fmt, cfg.max_patches, cfg.max_sents)
    val_ds = PatientDataset(cfg.data_root, "val", cfg.fmt, cfg.max_patches, cfg.max_sents)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Infer dims from one batch
    sample = next(iter(train_loader))
    patch_dim = sample["patches"].shape[-1]
    sent_dim = sample["sent_emb"].shape[-1]
    max_query_id = compute_max_query_id()

    model = PathEspresso(
        patch_dim=patch_dim,
        sent_dim=sent_dim,
        hidden=cfg.hidden,
        selector_proj=cfg.selector_proj,
        max_query_id=max_query_id,
        dropout=cfg.dropout,
    ).to(device)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler() if (cfg.fp16 and device.type == "cuda") else None

    use_wandb = bool(cfg and args.wandb_project) and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=cfg.__dict__,
        )

    best_val = float("inf")

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optim=optim,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
            use_wandb=use_wandb,
        )

        va = eval_one_epoch(model=model, loader=val_loader, device=device, cfg=cfg)

        logging.info(
            f"Epoch {epoch:03d} | train loss={tr['loss']:.4f} "
            f"(surv={tr['surv_loss']:.4f}, claim={tr['claim_loss']:.4f}, sent={tr['sent_ev_loss']:.4f}) "
            f"| val surv={va['surv_loss']:.4f} claim={va['claim_loss']:.4f} sent={va['sent_ev_loss']:.4f}"
        )

        if use_wandb:
            wandb.log({
                "val/surv_loss": va["surv_loss"],
                "val/claim_loss": va["claim_loss"],
                "val/sent_ev_loss": va["sent_ev_loss"],
                "val/sparsity_loss": va["sparsity_loss"],
                "epoch": epoch,
            })

        # Save last
        save_checkpoint(cfg.output_dir / "last.pt", model, optim, epoch, cfg)

        # Save best by survival loss (you can change criterion)
        if va["surv_loss"] < best_val:
            best_val = va["surv_loss"]
            save_checkpoint(cfg.output_dir / "best.pt", model, optim, epoch, cfg)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
