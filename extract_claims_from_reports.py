#!/usr/bin/env python3
"""
Extract Tier-1 claims + supporting sentence IDs from TCGA report_text.

Writes a new parquet/csv with:
- claim_targets_json: {"tumor_grade": 2, "margins": 1, ...}
- claim_sent_gt_json: {"tumor_grade": [3], "margins": [7,8], ...}

You can later train a sentence-grounding head using claim_sent_gt_json.

Usage:
  python extract_claims_from_reports.py \
    --meta_path meta/patients_all.parquet \
    --out_path meta/patients_all_with_claims.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger("extract_claims")

# -------------------------
# Sentence splitting
# -------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\;\:])\s+|\n+")

def split_sentences(text: str, max_sents: int = 512) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    if len(parts) > max_sents:
        parts = parts[:max_sents]
    return parts


# -------------------------
# Normalization helpers
# -------------------------
_ROMAN_MAP = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}

def roman_to_int(s: str) -> Optional[int]:
    s = (s or "").strip().lower()
    return _ROMAN_MAP.get(s)

def norm_bool_polarity(token: str) -> Optional[bool]:
    t = (token or "").strip().lower()
    neg = {"not identified", "not seen", "absent", "negative", "no", "free of", "clear", "uninvolved"}
    pos = {"present", "identified", "positive", "involved", "metastasis"}
    for k in neg:
        if k in t:
            return False
    for k in pos:
        if k in t:
            return True
    return None


# -------------------------
# Claim schema (class ids)
# -------------------------
# Keep it simple: unknown=0 always.

def encode_grade(n: Optional[int]) -> int:
    if n is None:
        return 0
    if n < 1 or n > 4:
        return 0
    return int(n)  # 1..4

def encode_yes_no_unknown(v: Optional[bool]) -> int:
    # 0=unknown, 1=no, 2=yes
    if v is None:
        return 0
    return 2 if v else 1

def encode_margins(v: Optional[bool]) -> int:
    # v=True -> positive/involved
    # v=False -> negative/free/clear
    if v is None:
        return 0
    return 2 if v else 1

def encode_size_bin(size_cm: Optional[float]) -> int:
    # 0=unk, 1<=2, 2=2-5, 3>5
    if size_cm is None or size_cm <= 0:
        return 0
    if size_cm <= 2.0:
        return 1
    if size_cm <= 5.0:
        return 2
    return 3

def encode_pT(pt: Optional[str]) -> int:
    # Minimal bucket example (expand later as needed)
    # 0=unk, 1=Tis, 2=T0, 3=T1, 4=T1a, 5=T1b, 6=T2, 7=T2a, 8=T2b, 9=T3, 10=T3a, 11=T4
    if not pt:
        return 0
    pt = pt.strip().lower().replace("pt", "t")
    mapping = {
        "tis": 1, "t0": 2, "t1": 3, "t1a": 4, "t1b": 5,
        "t2": 6, "t2a": 7, "t2b": 8,
        "t3": 9, "t3a": 10, "t4": 11,
    }
    return mapping.get(pt, 0)

def encode_pN(pn: Optional[str]) -> int:
    # 0=unk, 1=N0, 2=N1, 3=N2, 4=N3, 5=Nx
    if not pn:
        return 0
    pn = pn.strip().lower().replace("pn", "n")
    mapping = {"n0": 1, "n1": 2, "n2": 3, "n3": 4, "nx": 5}
    return mapping.get(pn, 0)


# -------------------------
# Regex patterns
# -------------------------
RE_FUHRMAN = re.compile(r"\bfuhrman\b.*?\bgrade\b\s*[:\-]?\s*([ivx]+)\s*/\s*([ivx]+)", re.I)
RE_NUCLEAR = re.compile(r"\bnuclear\s+grade\b\s*[:\-]?\s*([ivx]+)\s*/\s*([ivx]+)", re.I)
RE_GRADE_G = re.compile(r"\b(?:grade|g)\s*([1-4])\b", re.I)
RE_HIGHLOW_GRADE = re.compile(r"\b(high|low)\s+grade\b", re.I)

RE_TUMOR_TYPE = re.compile(r"\b(tumou?r\s*type)\s*:\s*(.+?)(?=\.|;|\n|$)", re.I)
RE_DIAGNOSIS_BLOCK = re.compile(
    r"\bdiagnosis\b\s*:(.*?)(?=\bgross\s+description\b|\bmicroscopic\s+description\b|\bintraoperative\b|$)",
    re.I | re.S,
)

RE_TUMOR_SIZE_1 = re.compile(r"\btumou?r\s+size\b.*?(?:is|:)\s*(?:greatest\s+diameter\s+is\s*)?([0-9]+(?:\.[0-9]+)?)\s*cm", re.I)
RE_TUMOR_SIZE_2 = re.compile(r"\bgreatest\s+dimension\b.*?([0-9]+(?:\.[0-9]+)?)\s*cm", re.I)
RE_TUMOR_DIMS = re.compile(r"\b(?:tumou?r|mass|lesion)\b.*?\b([0-9]+(?:\.[0-9]+)?)\s*x\s*([0-9]+(?:\.[0-9]+)?)\s*x\s*([0-9]+(?:\.[0-9]+)?)\s*cm", re.I)

RE_LOCAL_INV = re.compile(r"\blocal\s+invasion\b.*?\b(not\s+identified|not\s+seen|absent|negative|present|identified)\b", re.I)
RE_VASC_INV = re.compile(r"\b(lymphovascular\s+invasion|renal\s+vein\s+invasion|vascular\s+invasion|lvi)\b.*?\b(not\s+identified|not\s+seen|absent|negative|present|identified)\b", re.I)

RE_MARGINS = re.compile(r"\b(surgical\s+margins?|margins?)\b.*?\b(free\s+of\s+tumou?r|negative|clear|uninvolved|positive|involved)\b", re.I)

RE_PT = re.compile(r"\bpt\s*([0-4][a-c]?|is|x)\b", re.I)
RE_PN = re.compile(r"\bpn\s*([0-3]|x)\b|\bpn[0o]\b", re.I)  # handles pN0 / pNO


def find_first_match_sentence(sentences: List[str], pattern: re.Pattern) -> Tuple[Optional[re.Match], Optional[int]]:
    for i, s in enumerate(sentences):
        m = pattern.search(s)
        if m:
            return m, i
    return None, None


def extract_claims_from_report(report_text: str) -> Tuple[Dict[str, int], Dict[str, List[int]], List[str]]:
    sents = split_sentences(report_text)

    claim_targets: Dict[str, int] = {}
    claim_sent_gt: Dict[str, List[int]] = {}

    # ---- Tumor type (raw string -> you can map to normalized ids later)
    # For now: store unknown (0). If you want, add a dictionary mapper here.
    m_diag, sid_diag = find_first_match_sentence(sents, RE_TUMOR_TYPE)
    if not m_diag:
        # fallback: diagnosis block
        block = RE_DIAGNOSIS_BLOCK.search(report_text or "")
        if block:
            # set evidence as first sentence of diagnosis block if it appears in sents
            claim_targets["tumor_type"] = 0
            # not assigning sid reliably here unless you want to search a snippet
        else:
            claim_targets["tumor_type"] = 0
    else:
        claim_targets["tumor_type"] = 0
        claim_sent_gt["tumor_type"] = [int(sid_diag)] if sid_diag is not None else []

    # ---- Grade
    grade_val: Optional[int] = None
    grade_sid: Optional[int] = None

    m, sid = find_first_match_sentence(sents, RE_FUHRMAN)
    if m:
        g = roman_to_int(m.group(1))
        grade_val = g
        grade_sid = sid

    if grade_val is None:
        m, sid = find_first_match_sentence(sents, RE_NUCLEAR)
        if m:
            g = roman_to_int(m.group(1))
            grade_val = g
            grade_sid = sid

    if grade_val is None:
        m, sid = find_first_match_sentence(sents, RE_GRADE_G)
        if m:
            grade_val = int(m.group(1))
            grade_sid = sid

    claim_targets["tumor_grade"] = encode_grade(grade_val)
    if grade_sid is not None:
        claim_sent_gt["tumor_grade"] = [int(grade_sid)]

    # ---- Tumor size
    size_cm: Optional[float] = None
    size_sid: Optional[int] = None
    for pat in (RE_TUMOR_SIZE_1, RE_TUMOR_SIZE_2):
        m, sid = find_first_match_sentence(sents, pat)
        if m:
            size_cm = float(m.group(1))
            size_sid = sid
            break
    if size_cm is None:
        m, sid = find_first_match_sentence(sents, RE_TUMOR_DIMS)
        if m:
            a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
            size_cm = max(a, b, c)
            size_sid = sid

    claim_targets["tumor_size_bin"] = encode_size_bin(size_cm)
    if size_sid is not None:
        claim_sent_gt["tumor_size_bin"] = [int(size_sid)]

    # ---- Local invasion
    m, sid = find_first_match_sentence(sents, RE_LOCAL_INV)
    inv = norm_bool_polarity(m.group(1)) if m else None
    claim_targets["local_invasion"] = encode_yes_no_unknown(inv)
    if sid is not None:
        claim_sent_gt["local_invasion"] = [int(sid)]

    # ---- Vascular invasion
    m, sid = find_first_match_sentence(sents, RE_VASC_INV)
    vinv = norm_bool_polarity(m.group(2)) if m else None
    claim_targets["vascular_invasion"] = encode_yes_no_unknown(vinv)
    if sid is not None:
        claim_sent_gt["vascular_invasion"] = [int(sid)]

    # ---- Margins
    m, sid = find_first_match_sentence(sents, RE_MARGINS)
    # margin polarity: positive/involved => True, negative/free => False
    marg: Optional[bool] = None
    if m:
        token = m.group(2).lower()
        if "positive" in token or "involved" in token:
            marg = True
        elif "free of" in token or "negative" in token or "clear" in token or "uninvolved" in token:
            marg = False
    claim_targets["margins"] = encode_margins(marg)
    if sid is not None:
        claim_sent_gt["margins"] = [int(sid)]

    # ---- pT / pN
    m, sid = find_first_match_sentence(sents, RE_PT)
    pt = f"pt{m.group(1)}" if m else None
    claim_targets["pT"] = encode_pT(pt)
    if sid is not None:
        claim_sent_gt["pT"] = [int(sid)]

    m, sid = find_first_match_sentence(sents, RE_PN)
    pn = None
    if m:
        # Either group(1) from pn([0-3]|x) or matched pn0/pno token
        if m.group(1):
            pn = f"pn{m.group(1)}"
        else:
            pn = "pn0"
    claim_targets["pN"] = encode_pN(pn)
    if sid is not None:
        claim_sent_gt["pN"] = [int(sid)]

    return claim_targets, claim_sent_gt, sents


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(level=numeric, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--store_sentences", action="store_true", help="Store report_sentences_json for debugging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    meta_path = Path(args.meta_path)
    out_path = Path(args.out_path)

    df = pd.read_parquet(meta_path) if meta_path.suffix.lower() == ".parquet" else pd.read_csv(meta_path)

    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows)).copy()

    if "report_text" not in df.columns or "patient_id" not in df.columns:
        raise ValueError("meta must contain columns: patient_id, report_text")

    claim_targets_json = []
    claim_sent_gt_json = []
    report_sentences_json = []

    for _, row in df.iterrows():
        ct, cs, sents = extract_claims_from_report(str(row["report_text"]))
        claim_targets_json.append(json.dumps(ct, ensure_ascii=False, sort_keys=True))
        claim_sent_gt_json.append(json.dumps(cs, ensure_ascii=False, sort_keys=True))
        if args.store_sentences:
            report_sentences_json.append(json.dumps(sents, ensure_ascii=False))

    df["claim_targets_json"] = claim_targets_json
    df["claim_sent_gt_json"] = claim_sent_gt_json
    if args.store_sentences:
        df["report_sentences_json"] = report_sentences_json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(ou_
