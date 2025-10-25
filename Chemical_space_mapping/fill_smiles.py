
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fill SMILES & CAS (Flexible CSV)
--------------------------------
A robust variant that auto-detects CSV delimiter and header,
guesses name/SMILES/CAS columns, or accepts explicit overrides.

- Works with CSV or Excel.
- Auto-detects delimiter via csv.Sniffer unless --sep is provided.
- If no header, assumes the first column is compound names.
- Creates SMILES/CAS columns if missing.
- Same OPSIN + PubChem resolution pipeline.
- Optional tqdm progress bar and checkpointing.

Usage (CSV with a single Name column, e.g. 'CiCompoundList'):
  python fill_smiles_cas_flex.py --input CiCompoundList.csv \
    --name-col CiCompoundList --out CiCompoundList_filled.csv --progress

Dependencies: pandas, requests, rdkit, (optional) tqdm
"""

import argparse
import os
import re
import time
import json
import csv
from typing import Optional, Dict, Tuple, List

import pandas as pd
import requests
from rdkit import Chem

# Optional tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

CAS_REGEX = re.compile(r'^\s*\d{2,7}-\d{2}-\d\s*$')

def is_cas(s: str) -> bool:
    return bool(CAS_REGEX.match(str(s)))

def sanitize_smiles(smi: Optional[str]) -> Optional[str]:
    if not smi:
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m)  # canonicalize
    except Exception:
        return None

# ------------------ Web helpers ------------------
def opsin_from_name(name: str, timeout=10) -> Optional[str]:
    try:
        url = f"https://opsin.ch.cam.ac.uk/opsin/{requests.utils.quote(name)}.smi"
        r = requests.get(url, timeout=timeout)
        if r.ok:
            smi = r.text.strip()
            return sanitize_smiles(smi)
    except Exception:
        pass
    return None

def pubchem_cid_from_name(name: str, timeout=10) -> Optional[int]:
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON"
        r = requests.get(url, timeout=timeout)
        if r.ok:
            data = r.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception:
        pass
    return None

def pubchem_cid_from_smiles(smi: str, timeout=10) -> Optional[int]:
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"
        r = requests.post(url, data={"smiles": smi}, timeout=timeout)
        if r.ok:
            data = r.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception:
        pass
    return None

def pubchem_cid_from_cas(cas: str, timeout=10) -> Optional[int]:
    return pubchem_cid_from_name(cas, timeout=timeout)

def pubchem_smiles_from_cid(cid: int, timeout=10) -> Optional[str]:
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=timeout)
        if r.ok:
            smi = r.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return sanitize_smiles(smi)
    except Exception:
        pass
    return None

def pubchem_synonyms_from_cid(cid: int, timeout=10) -> Optional[list]:
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.json().get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
    except Exception:
        pass
    return None

def pick_cas_from_synonyms(syns: list, name_hint: Optional[str]=None) -> Optional[str]:
    if not syns:
        return None
    cas_list = [s for s in syns if is_cas(s)]
    if not cas_list:
        return None
    return cas_list[0]

# ------------------ Column handling ------------------
def guess_columns(df: pd.DataFrame, name_col: Optional[str], smiles_col: Optional[str], cas_col: Optional[str]) -> Tuple[str, str, str]:
    cols = list(df.columns)
    lc = {c.lower(): c for c in cols}

    def find(keys: List[str]) -> Optional[str]:
        for k in lc:
            if any(key in k for key in keys):
                return lc[k]
        return None

    nm = name_col or find(["name", "compound", "chemical", "cicomplist", "cicompoundlist", "cicompound"])
    smi = smiles_col or find(["smiles", "canonical_smiles", "structure"])
    cas = cas_col or find(["cas", "casrn", "cas_number", "cas#"])

    # If no name column can be guessed and there is exactly one column, assume it's names.
    if nm is None and len(cols) == 1:
        nm = cols[0]

    # Ensure SMILES/CAS columns exist
    if smi is None:
        smi = "SMILES"
        if "SMILES" not in df.columns:
            df["SMILES"] = ""
    if cas is None:
        cas = "CAS"
        if "CAS" not in df.columns:
            df["CAS"] = ""

    return nm, smi, cas

def load_table(input_path: str, sheet: Optional[str], sep: Optional[str], encoding: Optional[str]) -> pd.DataFrame:
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(input_path, sheet_name=sheet)
    # CSV: try sniffing if sep not given
    if sep:
        return pd.read_csv(input_path, sep=sep, encoding=encoding or "utf-8", engine="python")
    # auto-sniff
    with open(input_path, "r", encoding=encoding or "utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
            used_sep = dialect.delimiter
        except Exception:
            used_sep = ","
    return pd.read_csv(input_path, sep=used_sep, encoding=encoding or "utf-8", engine="python")

# ------------------ Resolution logic ------------------
def resolve_smiles(name: Optional[str], smiles: Optional[str], cas: Optional[str], sleep=0.12) -> Tuple[Optional[str], str]:
    smi = sanitize_smiles(smiles) if smiles else None
    if smi:
        return smi, "given"
    if name:
        smi = opsin_from_name(name)
        if smi:
            time.sleep(sleep)
            return smi, "opsin_from_name"
        cid = pubchem_cid_from_name(name)
        if cid:
            smi = pubchem_smiles_from_cid(cid)
            if smi:
                time.sleep(sleep)
                return smi, "pubchem_from_name"
    if cas:
        cid = pubchem_cid_from_cas(cas)
        if cid:
            smi = pubchem_smiles_from_cid(cid)
            if smi:
                time.sleep(sleep)
                return smi, "pubchem_from_cas"
    return None, "unresolved"

def resolve_cas(name: Optional[str], smiles: Optional[str], cas: Optional[str], sleep=0.12) -> Tuple[Optional[str], str]:
    if cas and is_cas(cas):
        return cas, "given"
    cid = None
    if smiles:
        cid = pubchem_cid_from_smiles(smiles)
    if cid is None and name:
        cid = pubchem_cid_from_name(name)
    if cid:
        syns = pubchem_synonyms_from_cid(cid)
        chosen = pick_cas_from_synonyms(syns, name_hint=name)
        if chosen:
            time.sleep(sleep)
            return chosen, "pubchem_synonym"
    return None, "unresolved"

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser(description="Fill missing SMILES and CAS (flexible CSV/Excel).")
    ap.add_argument("--input", required=True, help="Path to input CSV or XLSX.")
    ap.add_argument("--sheet", default=None, help="Sheet name for XLSX.")
    ap.add_argument("--sep", default=None, help="CSV delimiter override (e.g., ',', ';', '\\t', '|').")
    ap.add_argument("--encoding", default=None, help="File encoding (default utf-8).")
    ap.add_argument("--name-col", default=None, help="Column containing compound names.")
    ap.add_argument("--smiles-col", default=None, help="Column containing SMILES.")
    ap.add_argument("--cas-col", default=None, help="Column containing CAS numbers.")
    ap.add_argument("--out", required=True, help="Output CSV/XLSX path.")
    ap.add_argument("--cache", default=None, help="Optional JSON cache file to speed up repeated runs.")
    ap.add_argument("--no-web", action="store_true", help="Skip web lookups (only validate/canonicalize existing SMILES).")
    ap.add_argument("--sleep", type=float, default=0.12, help="Seconds to sleep between web calls.")
    ap.add_argument("--progress", action="store_true", help="Show a tqdm progress bar (pip install tqdm).")
    ap.add_argument("--verbose", action="store_true", help="Print per-row resolution details.")
    ap.add_argument("--every", type=int, default=50, help="Heartbeat print every N rows (if not verbose).")
    ap.add_argument("--checkpoint-out", default=None, help="Write periodic partial file to this path.")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="Rows between checkpoints (0 = disable).")
    args = ap.parse_args()

    df = load_table(args.input, args.sheet, args.sep, args.encoding)

    name_col, smiles_col, cas_col = guess_columns(df, args.name_col, args.smiles_col, args.cas_col)
    if name_col is None:
        raise SystemExit("Could not determine a name column. Use --name-col to specify it.")

    # Load cache
    cache: Dict[str, Dict[str, Optional[str]]] = {}
    if args.cache and os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    def cache_get(key: str) -> Tuple[Optional[str], Optional[str]]:
        hit = cache.get(key, {})
        return hit.get("smiles"), hit.get("cas")

    def cache_set(key: str, smiles: Optional[str], cas: Optional[str]):
        cache[key] = {"smiles": smiles, "cas": cas}

    total = len(df)
    print(f"Starting fill: {total} rows | input={args.input} | name_col='{name_col}' | smiles_col='{smiles_col}' | cas_col='{cas_col}'")

    # Counters
    cnt_smi_given = cnt_smi_opsin = cnt_smi_pub_name = cnt_smi_pub_cas = cnt_smi_unres = 0
    cnt_cas_given = cnt_cas_syn = cnt_cas_unres = 0

    iterator = range(total)
    if args.progress and tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="Resolving", unit="row")

    for i in iterator:
        row = df.iloc[i]
        nm = str(row[name_col]).strip() if pd.notna(row.get(name_col, "")) else ""
        smi0 = str(row[smiles_col]).strip() if pd.notna(row.get(smiles_col, "")) else ""
        cas0 = str(row[cas_col]).strip() if pd.notna(row.get(cas_col, "")) else ""

        cache_key = json.dumps({"n": nm, "s": smi0, "c": cas0}, sort_keys=True)
        cached_smi, cached_cas = cache_get(cache_key)

        smi_src = "cached" if cached_smi is not None else ""
        cas_src = "cached" if cached_cas is not None else ""

        if args.no_web:
            new_smi = sanitize_smiles(smi0) or smi0
            new_cas = cas0
            smi_src = smi_src or ("canon" if new_smi != smi0 else "given")
            cas_src = cas_src or ("given" if new_cas else "unresolved")
        else:
            if cached_smi is not None or cached_cas is not None:
                new_smi = cached_smi if cached_smi is not None else smi0
                new_cas = cached_cas if cached_cas is not None else cas0
            else:
                new_smi, smi_src = resolve_smiles(nm, smi0, cas0, sleep=args.sleep)
                new_cas, cas_src = resolve_cas(nm, new_smi or smi0, cas0, sleep=args.sleep)
                cache_set(cache_key, new_smi, new_cas)

        # Write back
        if new_smi:
            df.iat[i, df.columns.get_loc(smiles_col)] = new_smi
        if new_cas and is_cas(new_cas):
            df.iat[i, df.columns.get_loc(cas_col)] = new_cas

        # Counters
        if smi_src in ("given", "canon"):
            cnt_smi_given += 1
        elif smi_src == "opsin_from_name":
            cnt_smi_opsin += 1
        elif smi_src == "pubchem_from_name":
            cnt_smi_pub_name += 1
        elif smi_src == "pubchem_from_cas":
            cnt_smi_pub_cas += 1
        else:
            cnt_smi_unres += 1

        if cas_src == "given":
            cnt_cas_given += 1
        elif cas_src == "pubchem_synonym":
            cnt_cas_syn += 1
        else:
            cnt_cas_unres += 1

        if args.verbose:
            print(f"[{i+1}/{total}] name='{nm[:50]}' | SMILES:{smi_src} | CAS:{cas_src}")
        if not args.verbose and args.every > 0 and (i + 1) % args.every == 0:
            print(f"… processed {i+1}/{total} rows")

        if args.checkpoint_out and args.checkpoint_every > 0 and (i + 1) % args.checkpoint_every == 0:
            _save_partial(df, args)

    _save_final(df, args)

    if args.cache:
        try:
            json.dump(cache, open(args.cache, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception:
            pass

    print("\n=== Summary ===")
    print(f"SMILES: given/canon={cnt_smi_given}, opsin={cnt_smi_opsin}, pubchem(name)={cnt_smi_pub_name}, pubchem(cas)={cnt_smi_pub_cas}, unresolved={cnt_smi_unres}")
    print(f"CAS   : given={cnt_cas_given}, pubchem(synonym)={cnt_cas_syn}, unresolved={cnt_cas_unres}")
    print("Done.")

def _save_partial(df: pd.DataFrame, args):
    path = args.checkpoint_out
    if not path:
        return
    _write_df(df, path)
    print(f"[checkpoint] wrote partial to {path}")

def _save_final(df: pd.DataFrame, args):
    out = args.out
    _write_df(df, out)
    print(f"Wrote: {out}")

def _write_df(df: pd.DataFrame, path: str):
    out_ext = os.path.splitext(path)[1].lower()
    if out_ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        if out_ext == "":
            path = path + ".csv"
        df.to_csv(path, index=False)

if __name__ == "__main__":
    main()
