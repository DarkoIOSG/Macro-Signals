"""
fetch_signals.py — BTC Signals Dashboard Data Pipeline
=======================================================
Fetches all BTC on-chain/macro signals, computes ternary scores,
writes public/data.json, and sends Telegram alerts if thresholds are met.

Signal logic is ported directly from btc_combined_backtest.py.
Run daily via GitHub Actions (see .github/workflows/fetch_signals.yml).

Requirements:
  pip install yfinance pandas numpy requests fredapi
  CRYPTOQUANT_KEY, FRED_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID env vars
"""

import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# ============================================================
# 1. CONFIG CONSTANTS  — edit these freely
# ============================================================

DATA_JSON_PATH    = "public/data.json"
TRAIN_START       = "2015-01-01"   # full history for correct rolling percentiles
DAYS_TO_EXPORT    = 365            # days of history written to data.json

# Ternary scoring (match backtest exactly)
PERCENTILE_WINDOW = 730            # 2-year rolling window
PERCENTILE_HI     = 80             # above this percentile → +1 (bearish)
PERCENTILE_LO     = 20             # below this percentile → -1 (bullish)

# Alert thresholds — edit to taste
ALERT_COMPOSITE_BEARISH = 0.5     # composite >= this → bearish alert
ALERT_COMPOSITE_BULLISH = -0.5    # composite <= this → bullish alert
ALERT_BTC_CRASH_PCT     = -10.0   # BTC 7-day return <= this % → crash alert
ALERT_BTC_SURGE_PCT     = 15.0    # BTC 7-day return >= this % → surge alert

# API keys from environment
CRYPTOQUANT_KEY    = os.getenv("CRYPTOQUANT_KEY",    "")
FRED_API_KEY       = os.getenv("FRED_API_KEY",       "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

CQUANT_BASE = "https://api.cryptoquant.com/v1"


# ============================================================
# 2. SIGNAL METADATA REGISTRY
# ============================================================
# Each signal: label, category, description, direction, source
# direction "normal"  → high raw value = bearish (+1)
# direction "inverse" → high raw value = bullish (-1)

SIGNAL_META = {
    # --- CryptoQuant: Valuation ---
    "MVRV":          {"label": "MVRV",               "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Market Value to Realized Value. High = overvalued = bearish."},
    "SOPR":          {"label": "SOPR",               "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Spent Output Profit Ratio. Above 1 = coins moving in profit = bearish."},
    "SOPR_Adj":      {"label": "SOPR Adjusted",      "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Adjusted SOPR (excludes same-day transactions)."},
    "STH_SOPR":      {"label": "STH SOPR",           "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Short-Term Holder SOPR. High = recent buyers in profit."},
    "LTH_SOPR":      {"label": "LTH SOPR",           "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Long-Term Holder SOPR. High = long-term holders distributing."},
    "NVT":           {"label": "NVT Ratio",          "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Network Value to Transactions. High = overvalued vs on-chain activity."},
    "NVM":           {"label": "NVM Ratio",          "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Network Value to Metcalfe. High = overvalued vs network growth."},
    "S2F_Dev":       {"label": "S2F Deviation",      "category": "valuation",   "direction": "normal",  "source": "cryptoquant", "description": "Stock-to-Flow model deviation. High = price above model = bearish."},
    # --- CryptoQuant: On-Chain Activity ---
    "Exch_Reserve":  {"label": "Exchange Reserve",   "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "BTC held on exchanges (Binance). Rising = potential sell pressure."},
    "Exch_Netflow":  {"label": "Exchange Netflow",   "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "Net BTC inflow to exchanges. Positive = sell pressure = bearish."},
    "Whale_Ratio":   {"label": "Whale Ratio",        "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "Top-10 exchange inflows / total inflows. High = whale distribution."},
    "MPI":           {"label": "MPI",                "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "Miners' Position Index. High = miners selling = bearish."},
    "Puell":         {"label": "Puell Multiple",     "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "Daily miner revenue / 365-day MA. High = miners overselling = bearish."},
    "SOPR_Ratio":    {"label": "SOPR Ratio",         "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "LTH-SOPR / STH-SOPR ratio."},
    "Dormancy":      {"label": "Dormancy Flow",      "category": "onchain",     "direction": "normal",  "source": "cryptoquant", "description": "Average dormancy of spent outputs. High = old coins moving = bearish."},
    # --- CryptoQuant: Derivatives / Sentiment ---
    "Lev_Ratio":     {"label": "Leverage Ratio",     "category": "derivatives", "direction": "normal",  "source": "cryptoquant", "description": "Estimated leverage ratio on Binance. High = fragile market = bearish."},
    "SSR":           {"label": "SSR",                "category": "derivatives", "direction": "normal",  "source": "cryptoquant", "description": "Stablecoin Supply Ratio. High = less stablecoin buying power = bearish."},
    "Open_Interest": {"label": "Open Interest",      "category": "derivatives", "direction": "normal",  "source": "cryptoquant", "description": "Futures open interest (Binance). High = overheated = bearish."},
    "Coinbase_Prem": {"label": "Coinbase Premium",   "category": "derivatives", "direction": "inverse", "source": "cryptoquant", "description": "BTC premium on Coinbase vs Binance. Positive = US buying pressure = bullish."},
    "NRPL":          {"label": "NRPL",               "category": "derivatives", "direction": "normal",  "source": "cryptoquant", "description": "Net Realized Profit/Loss. Positive = profit-taking = bearish."},
    # --- Computed Proxies ---
    "MVRV_Proxy":    {"label": "Mayer Multiple",     "category": "proxy",       "direction": "normal",  "source": "computed",    "description": "Price / 200-day MA. Free proxy for MVRV. High = overvalued."},
    "Puell_Proxy":   {"label": "Puell Proxy",        "category": "proxy",       "direction": "normal",  "source": "computed",    "description": "14d price change / 365d price change. Miner revenue proxy."},
    "RealVol_30":    {"label": "Realized Vol 30d",   "category": "proxy",       "direction": "normal",  "source": "computed",    "description": "30-day annualized realized volatility. High = risk-off = bearish."},
    "RealVol_90":    {"label": "Realized Vol 90d",   "category": "proxy",       "direction": "normal",  "source": "computed",    "description": "90-day annualized realized volatility. High = risk-off = bearish."},
    "LR_1Y":         {"label": "1Y Log Return",      "category": "proxy",       "direction": "normal",  "source": "computed",    "description": "BTC 1-year log return. High = momentum / overbought = bearish."},
    "LR_2Y_Z":       {"label": "2Y Z-Score",         "category": "proxy",       "direction": "normal",  "source": "computed",    "description": "BTC 2-year price z-score. High = extended above long-run mean = bearish."},
    # --- Macro ---
    "VIX":           {"label": "VIX",                "category": "macro",       "direction": "inverse", "source": "yfinance",    "description": "CBOE Volatility Index. Spike = fear / capitulation = contrarian bullish."},
    "DXY":           {"label": "DXY",                "category": "macro",       "direction": "normal",  "source": "yfinance",    "description": "US Dollar Index. Strong USD = risk-off = bearish for BTC."},
    "SP500_Trend":   {"label": "S&P 500 Trend",      "category": "macro",       "direction": "inverse", "source": "yfinance",    "description": "S&P 500 % above/below 200-day MA. Positive trend = risk-on = bullish for BTC."},
    "Gold_90d":      {"label": "Gold 90d Return",    "category": "macro",       "direction": "normal",  "source": "yfinance",    "description": "Gold 90-day return. Rising gold = risk-off = bearish for BTC."},
    "HY_Spread":     {"label": "HY Credit Spread",   "category": "macro",       "direction": "normal",  "source": "fred",        "description": "High-yield credit spread (FRED: BAMLH0A0HYM2). High = credit stress = bearish."},
}


# ============================================================
# 3. DATA FETCHING
# ============================================================

def fetch_market_data():
    """Download BTC, SPY, VIX, DXY, Gold from yfinance. Returns clean Series."""

    def _dl(ticker, start=TRAIN_START):
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna().squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s

    print("  BTC-USD...",    end=" ", flush=True); close = _dl("BTC-USD"); print(f"{len(close)} days")
    print("  SPY...",        end=" ", flush=True); sp500 = _dl("SPY");     print(f"{len(sp500)} days")

    print("  ^VIX...",       end=" ", flush=True)
    try:
        vix = _dl("^VIX"); print(f"{len(vix)} days")
    except Exception:
        vix = None; print("FAILED")

    print("  DX-Y.NYB...",   end=" ", flush=True)
    try:
        dxy = _dl("DX-Y.NYB"); print(f"{len(dxy)} days")
    except Exception:
        dxy = None; print("FAILED")

    print("  GC=F (Gold)...", end=" ", flush=True)
    try:
        gold = _dl("GC=F"); print(f"{len(gold)} days")
    except Exception:
        gold = None; print("FAILED")

    return close, sp500, vix, dxy, gold


def fetch_fred_hy_spread(api_key: str):
    """Fetch HY credit spread from FRED. Returns pd.Series or None."""
    if not FRED_AVAILABLE or not api_key:
        print("  HY Spread: skipped (no fredapi or key)")
        return None
    try:
        fred = Fred(api_key=api_key)
        s = fred.get_series("BAMLH0A0HYM2", observation_start=TRAIN_START)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.dropna()
        print(f"  HY Spread (FRED): {len(s)} days")
        return s
    except Exception as e:
        print(f"  HY Spread FRED failed: {e}")
        return None


def _fetch_cquant_endpoint(endpoint: str, token: str, field: str,
                            exchange: str = None) -> "pd.Series | None":
    """Fetch one daily metric from CryptoQuant API v1."""
    params = {"window": "day", "from": TRAIN_START.replace("-", ""), "limit": 10000}
    if exchange:
        params["exchange"] = exchange
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"{CQUANT_BASE}/{endpoint}",
                         params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            return None
        rows = r.json().get("result", {}).get("data", [])
        if not rows:
            print("empty")
            return None
        df = pd.DataFrame(rows)
        if "date" not in df.columns or field not in df.columns:
            print(f"missing cols (have: {list(df.columns)[:5]})")
            return None
        df["date"] = pd.to_datetime(df["date"])
        s = pd.to_numeric(df.set_index("date")[field], errors="coerce")
        s.name = field
        return s.sort_index()
    except Exception as e:
        print(f"error: {e}")
        return None


def fetch_cryptoquant_signals(token: str) -> dict:
    """Fetch all 20 CryptoQuant on-chain signals. Returns {name: pd.Series}."""
    if not token:
        print("  CryptoQuant: skipped (no key)")
        return {}

    FETCHES = {
        # (endpoint, field, exchange_override)
        "MVRV":          ("btc/market-indicator/mvrv",                     "mvrv",                       None),
        "SOPR":          ("btc/market-indicator/sopr",                     "sopr",                       None),
        "SOPR_Adj":      ("btc/market-indicator/sopr",                     "a_sopr",                     None),
        "STH_SOPR":      ("btc/market-indicator/sopr",                     "sth_sopr",                   None),
        "LTH_SOPR":      ("btc/market-indicator/sopr",                     "lth_sopr",                   None),
        "NVT":           ("btc/network-indicator/nvt",                     "nvt",                        None),
        "NVM":           ("btc/network-indicator/nvm",                     "nvm",                        None),
        "S2F_Dev":       ("btc/network-indicator/stock-to-flow",           "stock_to_flow_reversion",    None),
        "Exch_Reserve":  ("btc/exchange-flows/reserve",                    "reserve",                    "binance"),
        "Exch_Netflow":  ("btc/exchange-flows/netflow",                    "netflow_total",               "binance"),
        "Whale_Ratio":   ("btc/flow-indicator/exchange-whale-ratio",       "exchange_whale_ratio",        "binance"),
        "MPI":           ("btc/flow-indicator/mpi",                        "mpi",                        None),
        "Puell":         ("btc/network-indicator/puell-multiple",          "puell_multiple",              None),
        "SOPR_Ratio":    ("btc/market-indicator/sopr-ratio",               "sopr_ratio",                  None),
        "Dormancy":      ("btc/network-indicator/dormancy",                "average_dormancy",            None),
        "Lev_Ratio":     ("btc/market-indicator/estimated-leverage-ratio", "estimated_leverage_ratio",    "binance"),
        "SSR":           ("btc/market-indicator/stablecoin-supply-ratio",  "stablecoin_supply_ratio",     None),
        "Open_Interest": ("btc/market-data/open-interest",                 "open_interest",               "binance"),
        "Coinbase_Prem": ("btc/market-data/coinbase-premium-index",        "coinbase_premium_index",      None),
        "NRPL":          ("btc/network-indicator/nrpl",                    "nrpl",                        None),
    }

    out = {}
    for name, (ep, fld, exch) in FETCHES.items():
        print(f"  CQ {name}...", end=" ", flush=True)
        s = _fetch_cquant_endpoint(ep, token, fld, exchange=exch)
        if s is not None:
            out[name] = s
            print(f"OK ({len(s)} pts)")
        else:
            print("FAILED (skipped)")
    return out


# ============================================================
# 4. SIGNAL COMPUTATION  (ported from btc_combined_backtest.py)
# ============================================================

def compute_proxy_signals(close: pd.Series, sp500: pd.Series,
                           vix, dxy, gold, hy) -> dict:
    """Compute proxy signals from public market data. No API key required."""
    out = {}
    idx = close.index
    log_ret = np.log(close / close.shift(1))

    # BTC-derived proxies
    ma200 = close.rolling(200, min_periods=100).mean()
    out["MVRV_Proxy"] = (close / ma200.replace(0, np.nan)).rename("MVRV_Proxy")

    ret14  = close.pct_change(14)
    ret365 = close.pct_change(365).replace(0, np.nan)
    out["Puell_Proxy"] = (ret14 / ret365).clip(-5, 5).rename("Puell_Proxy")

    out["RealVol_30"] = (log_ret.rolling(30, min_periods=10).std() * np.sqrt(365)).rename("RealVol_30")
    out["RealVol_90"] = (log_ret.rolling(90, min_periods=30).std() * np.sqrt(365)).rename("RealVol_90")
    out["LR_1Y"]      = np.log(close / close.shift(365)).rename("LR_1Y")

    lr      = np.log(close)
    lr_mean = lr.rolling(730, min_periods=180).mean()
    lr_std  = lr.rolling(730, min_periods=180).std().replace(0, np.nan)
    out["LR_2Y_Z"] = ((lr - lr_mean) / lr_std).rename("LR_2Y_Z")

    # Macro signals
    if vix is not None and len(vix) > 30:
        out["VIX"] = vix.reindex(idx, method="ffill").rename("VIX")

    if dxy is not None and len(dxy) > 30:
        out["DXY"] = dxy.reindex(idx, method="ffill").rename("DXY")

    if sp500 is not None and len(sp500) > 200:
        sp      = sp500.reindex(idx, method="ffill")
        sp_ma   = sp.rolling(200, min_periods=100).mean()
        out["SP500_Trend"] = ((sp - sp_ma) / sp_ma.replace(0, np.nan)).rename("SP500_Trend")

    if gold is not None and len(gold) > 90:
        g = gold.reindex(idx, method="ffill")
        out["Gold_90d"] = g.pct_change(90).rename("Gold_90d")

    if hy is not None and len(hy) > 30:
        out["HY_Spread"] = hy.reindex(idx, method="ffill").rename("HY_Spread")

    return out


def compute_ternary_scores(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw signal values to ternary {+1, 0, -1} using rolling percentiles.

    For each signal at date t:
      raw > 80th-pct rolling window → +1 (bearish, for normal direction)
      raw < 20th-pct rolling window → -1 (bullish, for normal direction)
      inverse direction: signs are flipped
    """
    result = pd.DataFrame(index=raw_df.index)
    min_p  = max(30, PERCENTILE_WINDOW // 4)   # need at least 25% of window filled

    for col in raw_df.columns:
        s         = raw_df[col].copy()
        direction = SIGNAL_META.get(col, {}).get("direction", "normal")

        p_hi = s.rolling(PERCENTILE_WINDOW, min_periods=min_p).quantile(PERCENTILE_HI / 100.0)
        p_lo = s.rolling(PERCENTILE_WINDOW, min_periods=min_p).quantile(PERCENTILE_LO / 100.0)

        ternary = pd.Series(0.0, index=s.index)
        valid   = s.notna() & p_hi.notna() & p_lo.notna()
        ternary[valid & (s > p_hi)] = +1.0
        ternary[valid & (s < p_lo)] = -1.0

        if direction == "inverse":
            ternary = -ternary

        result[col] = ternary

    return result


def compute_composite(ternary_df: pd.DataFrame) -> pd.Series:
    """Simple mean across all available ternary signals per day."""
    return ternary_df.mean(axis=1).rename("Composite")


# ============================================================
# 5. ALERT LOGIC
# ============================================================

def check_alerts(composite_today: float, btc_series: pd.Series) -> list:
    """Return list of alert message strings. Empty list = no alerts."""
    alerts = []

    if not math.isnan(composite_today):
        if composite_today >= ALERT_COMPOSITE_BEARISH:
            alerts.append(
                f"BEARISH SIGNAL: Composite score {composite_today:+.2f} "
                f"(threshold >= {ALERT_COMPOSITE_BEARISH})"
            )
        if composite_today <= ALERT_COMPOSITE_BULLISH:
            alerts.append(
                f"BULLISH SIGNAL: Composite score {composite_today:+.2f} "
                f"(threshold <= {ALERT_COMPOSITE_BULLISH})"
            )

    if len(btc_series) >= 7:
        btc_7d = btc_series.pct_change(7).iloc[-1] * 100
        if not math.isnan(btc_7d):
            if btc_7d <= ALERT_BTC_CRASH_PCT:
                alerts.append(
                    f"BTC CRASH WARNING: 7-day return {btc_7d:.1f}% "
                    f"(<= {ALERT_BTC_CRASH_PCT}%)"
                )
            if btc_7d >= ALERT_BTC_SURGE_PCT:
                alerts.append(
                    f"BTC SURGE ALERT: 7-day return {btc_7d:.1f}% "
                    f"(>= +{ALERT_BTC_SURGE_PCT}%)"
                )

    return alerts


def send_telegram_alert(alerts: list, composite: float, btc_price: float,
                         counts: dict, bearish_signals: list) -> None:
    """Send Telegram alert message. Silently skips if credentials missing."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  Telegram: skipped (no credentials)")
        return

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    alert_lines = "\n".join(f"  {a}" for a in alerts)

    top_bearish = ""
    if bearish_signals:
        top_bearish = f"\nTop bearish signals: {', '.join(bearish_signals[:7])}"

    message = (
        f"BTC SIGNAL ALERT - {today_str}\n\n"
        f"{alert_lines}\n\n"
        f"BTC Price:  ${btc_price:,.0f}\n"
        f"Composite:  {composite:+.2f}\n\n"
        f"Signal breakdown:\n"
        f"  Bullish: {counts['bullish']}  (-1)\n"
        f"  Neutral: {counts['neutral']}  ( 0)\n"
        f"  Bearish: {counts['bearish']}  (+1)\n"
        f"  Total:   {counts['total']}\n"
        f"{top_bearish}"
    )

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
        if r.status_code == 200:
            print("  Telegram alert sent.")
        else:
            print(f"  Telegram failed: HTTP {r.status_code} — {r.text[:200]}")
    except Exception as e:
        print(f"  Telegram error: {e}")


# ============================================================
# 6. JSON BUILDER
# ============================================================

def _to_float(v):
    """Convert numpy/pandas scalar to Python float, or None if NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _series_to_records(series: pd.Series, tail: int = DAYS_TO_EXPORT) -> list:
    """Convert a pd.Series to [{date, value}] records for the last `tail` days."""
    s = series.tail(tail)
    return [
        {"date": idx.strftime("%Y-%m-%d"), "value": _to_float(val)}
        for idx, val in s.items()
    ]


def build_data_json(close: pd.Series, raw_df: pd.DataFrame,
                     ternary_df: pd.DataFrame, composite: pd.Series,
                     alerts: list) -> dict:
    """Build the complete data.json dict."""

    # --- BTC price ---
    btc_price_records = _series_to_records(close)

    # --- Signals ---
    signals_out = {}
    export_index = close.index[-DAYS_TO_EXPORT:]

    for col in raw_df.columns:
        meta   = SIGNAL_META.get(col, {})
        raw_s  = raw_df[col].reindex(export_index)
        tern_s = ternary_df[col].reindex(export_index) if col in ternary_df.columns else pd.Series(np.nan, index=export_index)

        values = []
        for idx in export_index:
            r = _to_float(raw_s.get(idx))
            t = _to_float(tern_s.get(idx))
            values.append({
                "date":    idx.strftime("%Y-%m-%d"),
                "raw":     r,
                "ternary": int(t) if t is not None else None,
            })

        signals_out[col] = {
            "label":       meta.get("label", col),
            "category":    meta.get("category", "unknown"),
            "description": meta.get("description", ""),
            "direction":   meta.get("direction", "normal"),
            "source":      meta.get("source", "unknown"),
            "values":      values,
        }

    # --- Composite ---
    composite_records = _series_to_records(composite)

    # --- Summary ---
    today_ternary = ternary_df.iloc[-1]
    n_bullish = int((today_ternary == -1).sum())
    n_bearish = int((today_ternary ==  1).sum())
    n_neutral = int((today_ternary ==  0).sum())
    composite_today = _to_float(composite.iloc[-1])
    btc_today       = _to_float(close.iloc[-1])

    return {
        "last_updated":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "btc_price":       btc_price_records,
        "signals":         signals_out,
        "composite":       composite_records,
        "summary": {
            "composite_today":        composite_today,
            "btc_price_today":        btc_today,
            "bullish_signals":        n_bullish,
            "bearish_signals":        n_bearish,
            "neutral_signals":        n_neutral,
            "total_signals_available": len(ternary_df.columns),
            "alert_triggered":        len(alerts) > 0,
            "alert_message":          "; ".join(alerts) if alerts else None,
        },
    }


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print(f"\n{'='*60}")
    print(f"BTC Signals Pipeline — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # [1/6] Market data
    print("\n[1/6] Fetching market data from yfinance...")
    close, sp500, vix, dxy, gold = fetch_market_data()
    print(f"  BTC: {len(close)} days, latest {close.index[-1].date()} @ ${float(close.iloc[-1]):,.0f}")

    # [2/6] FRED
    print("\n[2/6] Fetching FRED HY Spread...")
    hy = fetch_fred_hy_spread(FRED_API_KEY)

    # [3/6] CryptoQuant
    print("\n[3/6] Fetching CryptoQuant on-chain signals...")
    cq_signals = fetch_cryptoquant_signals(CRYPTOQUANT_KEY)
    print(f"  {len(cq_signals)}/20 CryptoQuant signals fetched")

    # [4/6] Compute
    print("\n[4/6] Computing proxy signals and ternary scores...")
    proxy_signals = compute_proxy_signals(close, sp500, vix, dxy, gold, hy)
    print(f"  {len(proxy_signals)} proxy/macro signals computed")

    all_raw  = {**cq_signals, **proxy_signals}
    raw_df   = pd.DataFrame(all_raw).reindex(close.index).ffill()
    print(f"  Raw signal matrix: {raw_df.shape}")

    ternary_df = compute_ternary_scores(raw_df)
    composite  = compute_composite(ternary_df)

    today_ternary = ternary_df.iloc[-1]
    n_bullish = int((today_ternary == -1).sum())
    n_bearish = int((today_ternary ==  1).sum())
    n_neutral = int((today_ternary ==  0).sum())
    print(f"  Today: bullish={n_bullish}, neutral={n_neutral}, bearish={n_bearish}")
    print(f"  Composite today: {float(composite.iloc[-1]):+.4f}")

    # [5/6] Alerts
    print("\n[5/6] Checking alert thresholds...")
    composite_today = float(composite.iloc[-1])
    alerts = check_alerts(composite_today, close)

    if alerts:
        print(f"  {len(alerts)} alert(s) triggered:")
        for a in alerts:
            print(f"    - {a}")
        bearish_names = [col for col in ternary_df.columns if today_ternary.get(col, 0) == 1]
        send_telegram_alert(
            alerts=alerts,
            composite=composite_today,
            btc_price=float(close.iloc[-1]),
            counts={"bullish": n_bullish, "neutral": n_neutral,
                    "bearish": n_bearish, "total": len(ternary_df.columns)},
            bearish_signals=bearish_names,
        )
    else:
        print("  No alerts triggered.")

    # [6/6] Write JSON
    print("\n[6/6] Writing data.json...")
    data = build_data_json(close, raw_df, ternary_df, composite, alerts)

    out_path = DATA_JSON_PATH
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))  # compact, no indent for smaller file

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Written: {out_path} ({size_kb:.1f} KB)")
    print(f"  Signals: {len(data['signals'])}, Days: {DAYS_TO_EXPORT}")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
