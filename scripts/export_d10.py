#!/usr/bin/env python3
"""
export_d10.py — Convert D10 backtest CSV output to public/d10_data.json

Run after d10_production_backtest.py has finished:
  python scripts/d10_production_backtest.py   # writes /tmp/combo_d10_production.csv
  python scripts/export_d10.py               # writes public/d10_data.json

The dashboard will automatically load d10_data.json when present.
"""
import json
import os
import sys
from pathlib import Path

COMBO_CSV    = os.environ.get("D10_COMBO_CSV",    "/tmp/combo_d10_production.csv")
EXPOSURE_CSV = os.environ.get("D10_EXPOSURE_CSV", "/tmp/exposure_d10_production.csv")
OUT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "public", "d10_data.json")
DAYS_TO_KEEP = int(os.environ.get("D10_DAYS", "730"))  # 2 years by default


def load_series(csv_path: str) -> list[dict]:
    """Read a single-column CSV (index=date, col=value) and return [{date, value}]."""
    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.dropna()
    # Keep last N days
    df = df.iloc[-DAYS_TO_KEEP:]
    col = df.columns[0]
    return [{"date": idx.strftime("%Y-%m-%d"), "value": round(float(row[col]), 6)}
            for idx, row in df.iterrows()]


def main():
    missing = [p for p in (COMBO_CSV, EXPOSURE_CSV) if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Missing input files: {missing}")
        print("Run d10_production_backtest.py first to generate the CSV outputs.")
        sys.exit(1)

    print(f"Loading combo from  {COMBO_CSV}")
    combo = load_series(COMBO_CSV)

    print(f"Loading exposure from {EXPOSURE_CSV}")
    exposure = load_series(EXPOSURE_CSV)

    # Latest values
    last_combo    = combo[-1]["value"]    if combo    else None
    last_exposure = exposure[-1]["value"] if exposure else None
    last_date     = combo[-1]["date"]     if combo    else None

    out = {
        "last_updated": last_date,
        "combo":        combo,
        "exposure":     exposure,
        "summary": {
            "combo_today":    last_combo,
            "exposure_today": last_exposure,
        }
    }

    out_path = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"Written {len(combo)} combo + {len(exposure)} exposure rows → {out_path}")
    print(f"Latest: combo={last_combo:.4f}, exposure={last_exposure:.1f}%")


if __name__ == "__main__":
    main()
