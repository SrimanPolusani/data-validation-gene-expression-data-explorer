"""Run integrity checks on the loaded GSE183947 data and save a short report."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_dataset

logger = logging.getLogger(__name__)

REPORT_FILE = Path(__file__).resolve().parent.parent / "reports" / "validation_report.md"
results = []  # (section, status, check, detail)


def check(section, ok, name, detail="", warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    results.append((section, status, name, detail))
    logger.info("%s  %s %s", status, name, detail)


def validate(expr, meta):
    s = "Data structure"
    check(s, expr.shape == (20246, 60), "Matrix is 20,246 genes x 60 samples", str(expr.shape))
    check(s, all(expr.dtypes == "float64"), "All values are numeric")
    check(s, not expr.isna().any().any(), "No missing values")
    check(s, not (expr < 0).any().any(), "No negative FPKM values")
    check(s, list(expr.columns) == list(meta.index), "Samples in the same order in both tables")

    s = "Sample annotations"
    counts = meta["condition"].value_counts()
    check(s, counts.get("tumor") == 30 and counts.get("normal") == 30, "30 tumor / 30 normal")
    # CA.<donor> is a tumor sample, CAP.<donor> the matched normal tissue
    label = meta.index.str.split(".").str[0].map({"CA": "tumor", "CAP": "normal"})
    check(s, (label == meta["condition"]).all(), "Condition matches CA/CAP sample name")
    check(s, (meta["title"].str.split().str[0] == meta["condition"]).all(),
          "Condition matches GEO sample title")
    check(s, (meta.index.str.split(".").str[1] == meta["donor"]).all(),
          "Donor ID in sample name matches donor field")
    check(s, meta["geo_accession"].str.match(r"GSM\d+$").all() and meta["geo_accession"].is_unique,
          "GEO accessions are valid and unique")

    pairs = pd.crosstab(meta["donor"], meta["condition"]).gt(0).all(axis=1)
    unpaired = meta.index[meta["donor"].isin(pairs[~pairs].index)]
    check(s, pairs.all(), "Every donor has a tumor/normal pair",
          f"{pairs.sum()} pairs; unpaired: {', '.join(unpaired)}", warn_only=True)
    mixed = meta.groupby("donor")["metastasis"].nunique()
    check(s, (mixed == 1).all(), "Metastasis label agrees within each donor",
          f"conflict: donor {', '.join(mixed[mixed > 1].index)}", warn_only=True)

    s = "Gene identifiers"
    genes = expr.index.to_series()
    check(s, genes.is_unique, "Gene names are unique")
    check(s, (genes.str.strip() == genes).all() and (genes != "").all(),
          "No blank names or stray spaces")
    check(s, not genes.str.startswith("ENSG").any(), "One identifier type (gene symbols)")
    dates = [g for g in genes if re.match(r"\d{1,2}-(Mar|Sep|Dec)$", g)]
    check(s, not dates, "No gene names turned into dates by Excel",
          f"{len(dates)} found, e.g. {', '.join(dates[:4])}", warn_only=True)
    never = int((expr == 0).all(axis=1).sum())
    check(s, never == 0, "Every gene detected in at least one sample",
          f"{never} genes are always zero", warn_only=True)

    s = "Sample quality"
    detected = (expr > 0).sum()
    low = detected[detected < 0.7 * detected.median()]
    check(s, low.empty, "All samples detect >= 70% of the median gene count",
          f"median {detected.median():,.0f}; low: " +
          ", ".join(f"{k} ({v:,})" for k, v in low.items()), warn_only=True)

    # average correlation of each sample with all the others, on log2(FPKM + 1)
    corr = np.log2(expr + 1).corr().to_numpy(copy=True)
    np.fill_diagonal(corr, np.nan)
    mean_r = pd.Series(np.nanmean(corr, axis=0), index=expr.columns)
    odd = mean_r[mean_r < 0.7].sort_values()
    check(s, odd.empty, "All samples correlate with the cohort (mean r >= 0.7)",
          f"median r {mean_r.median():.2f}; low: " +
          ", ".join(f"{k} ({v:.2f})" for k, v in odd.items()), warn_only=True)


def write_report(path):
    lines = ["# Data validation report: GSE183947", ""]
    for status in ("PASS", "WARN", "FAIL"):
        lines.append(f"- {status}: {sum(r[1] == status for r in results)}")
    for section in dict.fromkeys(r[0] for r in results):
        lines += ["", f"## {section}", "", "| Status | Check | Detail |", "|---|---|---|"]
        lines += [f"| {st} | {name} | {d} |" for sec, st, name, d in results if sec == section]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    expr, meta = load_dataset()
    validate(expr, meta)
    write_report(REPORT_FILE)

    fails = sum(r[1] == "FAIL" for r in results)
    print(f"\n{len(results)} checks, {fails} failed. Report saved to {REPORT_FILE}")
    raise SystemExit(1 if fails else 0)
