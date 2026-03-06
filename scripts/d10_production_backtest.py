#!/usr/bin/env python3
"""
D10 Production Strategy â Final Backtest
=========================================
D10 | 2Y+3Y Ensemble | H90 | Original MVRV | Holm Gate | Uniform Horizon Weights
Sharpe: ~1.179 (2020-01-01 to present)

Depends on: btc_combined_backtest.py (core library, do not modify)

Parameters (all principled or design choices):
  - Delta: 10% (11 tiers, 5 thresholds)
  - Pair selection horizon: H90 (90-day forward returns for pair_power)
      Principled: predictive power monotonically increases with horizon
      (H3=53% HR, H30=54%, H90=58%, H180=62%). Selecting on H90 finds
      pairs with the strongest genuine signal.
  - Inference scoring weights: UNIFORM across all 6 horizons (3/7/14/30/90/180)
      Principled: maximum entropy / null hypothesis. No a priori reason to
      weight any horizon more than another at inference. Prior weights
      {H30=0.30, H90=0.25} were inherited from btc_regime_backtest.py with
      no documented rationale â effectively an undocumented parameter choice
      made without human review. Uniform is the only zero-parameter option.
      Empirically: Uniform Sharpe 1.179 vs original 1.120 (p=0.30, not
      significant) â but uniform is adopted on principled grounds, not
      because it backtests better.
  - Training windows: 2Y + 3Y ensemble (average combo scores)
  - MVRV regime: Fixed boundaries (COLD<0, NEUTRAL<2, HOT<5, EXTREMEâ¥5)
  - Regime weights: same=1.0, adjacent=0.5, distant=0.1
  - Gate: Holm step-down sequential test (90d horizon, alpha=0.05)
  - Thresholds: Adaptive OOS percentiles (P17/P33/P50/P67/P83)
  - Top-N: Top 1% of pair universe (ceil(n_pairs * 0.01)) with mean+1Ï floor, min 5
  - Cooldown: 7 days (directional change lockout)
  - Rebalance: Daily signal reaction, quarterly pair re-selection
  - Gate/threshold lookback: 3Y on ensemble combo output

Changelog:
  2026-03-04: Initial production version. H30=0.30 weights inherited.
  2026-03-05: Switched to uniform horizon weights at inference (Momir decision).
              Rationale: prior weights undocumented, uniform is null hypothesis.
              MABreak lookahead fix pending (one-line guard in _find_events).

Author: Mac (OpenClaw agent) + Momir
Date: March 4â5, 2026
"""

import sys
import os
import math
import warnings
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from itertools import combinations

warnings.filterwarnings('ignore')

# ââ Configuration ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
TRAIN_START = "2014-01-01"
EVAL_START = "2020-01-01"
ENSEMBLE_WINDOWS = [2, 3]          # Training window lengths (years) for ensemble
GATE_THRESHOLD_LOOKBACK = 3        # Years of ensemble combo to use for gate/threshold calibration
PRIMARY_HORIZON = 90               # Pair scoring horizon (days)
DELTA = 0.10                       # Allocation step size (10%)
N_THRESHOLDS = 5                   # Number of adaptive percentile thresholds
COOLDOWN_DAYS = 7                  # Directional change lockout
TOP_N_PCT = 0.01                   # Top 1% of pair universe
MIN_PAIRS = 5                      # Minimum pairs (fallback if floor filters too many)
SIG_FLOOR_SIGMA = 1.0              # Significance floor: mean + N*sigma
ACTIVITY_THRESHOLD = 0.05          # Minimum signal activity rate
TX_COST_BPS = 10                   # Transaction cost per unit of exposure change (bps)
CRYPTOQUANT_KEY = os.environ.get("CRYPTOQUANT_KEY", "")

# ââ Load core library ââââââââââââââââââââââââââââââââââââââââââââââââââââ
LIB_PATH = os.environ.get(
    "BTC_LIB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_combined_backtest_lib.py")
)
spec = importlib.util.spec_from_file_location("bbt", LIB_PATH)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# ââ Override inference horizon weights with uniform (null hypothesis) âââââ
# The library default {H30=0.30, H90=0.25} was an undocumented inherited choice.
# Uniform is the maximum-entropy null: no a priori reason to weight any horizon
# over another at inference. Pair SELECTION still uses H90 (pair_power), which
# is principled because predictive power increases monotonically with horizon.
_mod.HORIZON_WEIGHTS.clear()
_mod.HORIZON_WEIGHTS.update({3: 1/6, 7: 1/6, 14: 1/6, 30: 1/6, 90: 1/6, 180: 1/6})

# ââ Fix #1: MABreak lookahead bias (colleague-reported, confirmed real) ââ
# _find_events computes forward returns without checking that the endpoint
# falls before the evaluation date. This leaks future data into training.
# Bug affects 6 of 55 signals (MABreak variants). One-line guard.
# Colleague's "fixed" file had this guard COMMENTED OUT â not actually applied.
_orig_find_events = _mod.MABreakSignal._find_events

def _patched_find_events(self, brk_df, close, bit, dirn, holding, before, horizons):
    import numpy as np
    m = ((brk_df["bit"] == bit) & (brk_df["direction"] == dirn)
         & (brk_df["date"] < before))
    cands = brk_df[m].copy()
    if cands.empty:
        return pd.DataFrame()
    for k in self._related_keys(bit, list(holding.keys())):
        if cands.empty:
            break
        if k in holding and k != bit:
            try:
                cands = cands[cands["holding"].apply(
                    lambda h: isinstance(h, dict) and h.get(k) == holding[k])]
            except Exception:
                break
    if cands.empty:
        return pd.DataFrame()
    clustered, ld = [], None
    for _, r in cands.iterrows():
        if ld is None or (r["date"] - ld).days >= _mod.CLUSTER_GAP:
            clustered.append(r)
            ld = r["date"]
    cands = pd.DataFrame(clustered)
    if cands.empty:
        return pd.DataFrame()
    evts = []
    for _, r in cands.iterrows():
        d = r["date"]
        if d not in close.index:
            continue
        p0 = close.loc[d]
        rec = {"date": d, "bit": bit, "direction": dirn, "price": p0}
        for h in horizons:
            fi = close.index.searchsorted(d) + h
            if fi < len(close) and close.index[fi] < before:  # â THE FIX
                rec[f"fwd_{h}d"] = close.iloc[fi] / p0 - 1
            else:
                rec[f"fwd_{h}d"] = np.nan
        evts.append(rec)
    return pd.DataFrame(evts)

_mod.MABreakSignal._find_events = _patched_find_events

import yfinance as yf


def fetch_data():
    """Fetch all required data: BTC price, macro proxies, CQ signals, technicals."""
    close = yf.download("BTC-USD", start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    def _dl(ticker):
        try:
            r = yf.download(ticker, start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
            r.index = pd.to_datetime(r.index).tz_localize(None)
            return r.reindex(close.index, method="ffill")
        except Exception:
            return pd.Series(np.nan, index=close.index)

    sp500 = _dl("^GSPC")
    vix = _dl("^VIX")
    dxy = _dl("DX-Y.NYB")
    gold = _dl("GC=F")
    hy = _dl("HYG")

    # CryptoQuant on-chain signals
    cq = _mod.fetch_cquant_signals(CRYPTOQUANT_KEY, start=TRAIN_START)
    # Proxy signals
    prx = _mod.compute_proxy_signals(close, sp500, vix, dxy, gold, hy)
    # Raw signal dataframe
    raw_df = pd.DataFrame({**cq, **prx}).reindex(close.index).ffill()
    # Ternary matrix
    ter = _mod.compute_ternary_matrix(raw_df)
    # Technical signals
    tsr = _mod.compute_technical_signals(close, sp500, _mod.build_technical_signal_registry())
    # Combined
    combined_df = pd.concat([ter, tsr], axis=1).reindex(close.index).fillna(0)
    # MVRV for regime weighting
    mvrv_raw = raw_df.get("MVRV")

    return close, combined_df, mvrv_raw, sp500


def prepare_signals(close, combined_df):
    """Discretize signals and compute forward returns.
    FIX 4: activity filtering moved into build_ensemble_combo per training window
    to avoid full-sample lookahead in signal candidate selection.
    """
    disc = _mod.discretize_signals(combined_df)
    fwd  = _mod.compute_forward_returns(close)
    print(f"Signals: {len(combined_df.columns)} total")
    return disc, fwd


def find_gate_holm(combo_trail, close_trail, horizon=PRIMARY_HORIZON, alpha=0.05):
    """Holm step-down sequential test for gate selection."""
    cn = combo_trail[combo_trail != 0].dropna()
    if len(cn) < 60:
        return 0.0
    fwd_r = close_trail.pct_change(horizon).shift(-horizon)
    com = cn.index.intersection(fwd_r.dropna().index)
    if len(com) < 30:
        return 0.0
    c = cn.loc[com]
    f = fwd_r.loc[com]
    hit = ((c < 0) & (f > 0)) | ((c > 0) & (f < 0))

    for step, pct in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], 1):
        thr = 0.0 if pct == 0 else np.percentile(c.abs(), pct)
        mask = c.abs() >= thr if pct > 0 else pd.Series(True, index=c.index)
        n = mask.sum()
        if n < 20:
            continue
        k = int(hit[mask].sum())
        if binomtest(k, n, 0.5, alternative='greater').pvalue < alpha / step:
            return thr
    return float(np.percentile(c.abs(), 50))


def apply_cooldown(raw_exposure, cd=COOLDOWN_DAYS):
    """Apply directional change cooldown (Fix 1 from Darko: track last_attempted, not last_executed)."""
    res = raw_exposure.copy()
    prev = raw_exposure.iloc[0]
    # FIX 1 (from Darko): track every attempted direction, not just executed ones.
    # v1 bug: blocked reversals left last_dir stale, allowing bypass on the next day.
    last_attempted = 0
    last_change_date = raw_exposure.index[0]

    for i, (dt, tgt) in enumerate(raw_exposure.items()):
        if i == 0:
            res[dt] = tgt
            prev = tgt
            continue
        d = tgt - prev
        if abs(d) < 0.001:
            res[dt] = prev
            continue
        dirn = 1 if d > 0 else -1
        # FIX 1: update attempted direction BEFORE the block check
        is_rev = (last_attempted != 0) and (dirn != last_attempted)
        last_attempted = dirn

        if is_rev and (dt - last_change_date).days < cd:
            res[dt] = prev
        else:
            res[dt] = tgt
            prev = tgt
            last_change_date = dt
    return res


def select_pairs_with_floor(pairs, disc_tr, fwd_tr, rd, weights, top_n):
    """Select top 1% pairs with mean+1Ï significance floor."""
    meta = {
        (s1, s2): _mod.pair_power(disc_tr, fwd_tr, s1, s2, rd, weights=weights, primary_horizon=PRIMARY_HORIZON)
        for s1, s2 in pairs
    }
    ranked = sorted(meta.items(), key=lambda x: -x[1])
    powers = np.array([s for _, s in ranked])

    # Significance floor: mean + 1Ï
    floor = np.mean(powers) + SIG_FLOOR_SIGMA * np.std(powers)

    # Take top pairs that pass floor, up to top_n
    selected = []
    for pair, score in ranked:
        if score < floor:
            break
        selected.append(pair)
        if len(selected) >= top_n:
            break

    # Fallback: if fewer than MIN_PAIRS pass, take top MIN_PAIRS
    if len(selected) < MIN_PAIRS:
        selected = [p for p, _ in ranked[:MIN_PAIRS]]
        method = f"fallback_{MIN_PAIRS}"
    else:
        method = f"floor_{len(selected)}"

    return selected, method, floor


def build_ensemble_combo(close, disc, fwd, mvrv_aligned):
    """Build 2Y+3Y ensemble combo signal with walk-forward pair selection.

    FIX 4: candidate signal universe and top_n are re-computed inside each
    training window to avoid full-sample activity lookahead bias.
    """
    rebal_dates = pd.date_range(pd.Timestamp(EVAL_START), close.index[-1], freq="3MS")
    sigs_all = list(disc.columns)

    combo_parts = {}
    for ty in ENSEMBLE_WINDOWS:
        combo = pd.Series(0.0, index=close.index)
        print(f"\n  Building {ty}Y combo...")

        for i, rd in enumerate(rebal_dates):
            train_start = rd - pd.DateOffset(years=ty)
            next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else close.index[-1] + pd.Timedelta(days=1)

            in_train = (disc.index >= train_start) & (disc.index < rd)
            disc_tr = disc[in_train]
            fwd_tr = fwd[in_train]
            mvrv_tr = mvrv_aligned[in_train]

            # FIX 4: per-window activity filter â prevents full-sample lookahead
            act_tr         = {s: (disc_tr[s] != 0).mean() for s in sigs_all}
            active_sigs_tr = [s for s, a in act_tr.items() if a > ACTIVITY_THRESHOLD]
            pairs_tr       = list(combinations(active_sigs_tr, 2))
            top_n_tr       = max(math.ceil(len(pairs_tr) * TOP_N_PCT), MIN_PAIRS)

            # MVRV regime weighting (original fixed boundaries)
            mvrv_at_rd = float(mvrv_aligned.asof(rd)) if not pd.isna(mvrv_aligned.asof(rd)) else np.nan
            regime = _mod._get_regime(mvrv_at_rd)
            n_same = int((mvrv_tr.apply(lambda v: _mod._get_regime(v) == regime)).sum())
            weights = _mod._regime_weights(mvrv_tr, regime) if n_same >= _mod.MIN_REGIME_TRAIN_DAYS else None

            # Select pairs with significance floor
            top_pairs, method, floor = select_pairs_with_floor(pairs_tr, disc_tr, fwd_tr, rd, weights, top_n_tr)

            if i % 8 == 0:
                print(f"    [{rd.date()}] {method} (regime={regime}, n_sigs={len(active_sigs_tr)}, floor={floor:.3f})")

            # Score OOS dates
            oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
            for t in oos_dates:
                combo.loc[t] = _mod.score_at_date(disc, fwd, top_pairs, t)

        combo_parts[ty] = combo

    # Average ensemble
    ensemble = sum(combo_parts.values()) / len(combo_parts)
    return ensemble


def run_d10_backtest(combo, close):
    """Run D10 allocation backtest with Holm gate and adaptive thresholds."""
    rebal_dates = pd.date_range(pd.Timestamp(EVAL_START), close.index[-1], freq="3MS")
    ty = GATE_THRESHOLD_LOOKBACK

    # Compute gates at each rebalance
    gates = {}
    for rd in rebal_dates:
        combo_trail = combo[(combo.index >= rd - pd.DateOffset(years=ty)) & (combo.index < rd)]
        close_trail = close[(close.index >= rd - pd.DateOffset(years=ty)) & (close.index < rd)]
        gates[rd] = find_gate_holm(combo_trail, close_trail)

    # Build exposure series
    exposure = pd.Series(0.5, index=combo.index)
    prev = 0.5

    for qi, rd in enumerate(rebal_dates):
        next_rd = rebal_dates[qi + 1] if qi + 1 < len(rebal_dates) else close.index[-1] + pd.Timedelta(days=1)

        # Adaptive thresholds from trailing non-zero |combo|
        hist_nz = combo[(combo.index >= rd - pd.DateOffset(years=ty)) & (combo.index < rd)]
        hist_nz = hist_nz[hist_nz != 0].abs().dropna()

        if len(hist_nz) >= 30:
            thresholds = [np.percentile(hist_nz, 100 * i / (N_THRESHOLDS + 1)) for i in range(1, N_THRESHOLDS + 1)]
        else:
            thresholds = [(i + 1) / (N_THRESHOLDS + 1) for i in range(N_THRESHOLDS)]

        gate = gates.get(rd, 0.0)

        # Apply signal to OOS dates
        for t in combo.index[(combo.index >= rd) & (combo.index < next_rd)]:
            c = combo.loc[t]
            if np.isnan(c) or abs(c) < gate:
                exposure.loc[t] = prev
                continue
            n_above = sum(abs(c) >= th for th in thresholds)
            if c < 0:  # Bullish (negative combo = bullish)
                prev = min(0.5 + n_above * DELTA, 1.0)
            else:       # Bearish (positive combo = bearish)
                prev = max(0.5 - n_above * DELTA, 0.0)
            exposure.loc[t] = prev

    return apply_cooldown(exposure)


def compute_metrics(close_eval, exposure, name):
    """Compute geometric Sharpe and other performance metrics."""
    exp = exposure.reindex(close_eval.index).ffill()
    bt = _mod.backtest_from_exposure(close_eval, exp, name)
    r = bt['port_ret']

    total_return = (1 + r).prod() - 1
    n_years = len(r) / 365.25
    ann_ret = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = r.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    # Holdout Sharpe (2024-2025)
    ho_r = r[r.index.year.isin([2024, 2025])]
    if len(ho_r) > 30:
        ho_tr = (1 + ho_r).prod() - 1
        ho_ny = len(ho_r) / 365.25
        ho_ar = (1 + ho_tr) ** (1 / ho_ny) - 1
        ho_av = ho_r.std() * np.sqrt(365)
        ho_sharpe = ho_ar / ho_av if ho_av > 0 else 0
    else:
        ho_sharpe = 0

    # Yearly returns
    yearly = {yr: (1 + r[r.index.year == yr]).prod() - 1 for yr in sorted(r.index.year.unique())}

    # Fees
    fees = exp.diff().abs().fillna(0).sum() * TX_COST_BPS / n_years

    return {
        'name': name, 'sharpe': sharpe, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'max_dd': max_dd, 'ho_sharpe': ho_sharpe, 'yearly': yearly, 'fees': fees,
        'avg_exposure': exp.mean()
    }


def main():
    print("=" * 70)
    print("D10 PRODUCTION STRATEGY BACKTEST")
    print("2Y+3Y Ensemble | H90 | Original MVRV | Holm Gate")
    print("=" * 70)

    # Fetch data
    print("\n[1/5] Fetching data...")
    close, combined_df, mvrv_raw, sp500 = fetch_data()
    mvrv_aligned = mvrv_raw.reindex(close.index, method="ffill")

    # Prepare signals
    print("\n[2/5] Preparing signals...")
    disc, fwd = prepare_signals(close, combined_df)

    # Build ensemble combo
    print("\n[3/5] Building 2Y+3Y ensemble combo (H90, MVRV-weighted)...")
    combo = build_ensemble_combo(close, disc, fwd, mvrv_aligned)

    # Run backtest
    print("\n[4/5] Running D10 backtest...")
    exposure = run_d10_backtest(combo, close)
    exposure_eval = exposure.loc[EVAL_START:]

    # Compute metrics
    print("\n[5/5] Computing metrics...")
    close_eval = close.loc[EVAL_START:]

    m_strat = compute_metrics(close_eval, exposure_eval, "D10 Production")
    m_bh = compute_metrics(close_eval, pd.Series(1.0, index=close_eval.index), "BTC B&H")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for m in [m_strat, m_bh]:
        print(f"\n  {m['name']}:")
        print(f"    Sharpe:          {m['sharpe']:.4f}")
        print(f"    Ann Return:      {m['ann_ret']:+.1%}")
        print(f"    Ann Vol:         {m['ann_vol']:.1%}")
        print(f"    Max Drawdown:    {m['max_dd']:.1%}")
        print(f"    HO Sharpe:       {m['ho_sharpe']:.4f}")
        print(f"    Avg Exposure:    {m['avg_exposure']:.1%}")
        print(f"    Fees (bp/yr):    {m['fees']:.1f}")
        print(f"    Yearly:")
        for yr, ret in m['yearly'].items():
            print(f"      {yr}: {ret:+.1%}")

    # Save combo and exposure
    combo.to_csv("/tmp/combo_d10_production.csv")
    exposure_eval.to_csv("/tmp/exposure_d10_production.csv")
    print(f"\n  Combo saved to /tmp/combo_d10_production.csv")
    print(f"  Exposure saved to /tmp/exposure_d10_production.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
