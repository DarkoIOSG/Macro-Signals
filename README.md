# BTC Signals Dashboard

Zero-cost daily BTC signals dashboard. GitHub Actions fetches data, Vercel serves the React app.

## Architecture

```
GitHub Actions (cron: daily 8am UTC)
  → scripts/fetch_signals.py
  → public/data.json (committed to repo)

Vercel
  → React app fetches /data.json on load
  → Renders charts and signal table
```

## Setup

### 1. Create GitHub repo and push this folder

```bash
cd signals-dashboard
git init
git add .
git commit -m "initial"
git remote add origin https://github.com/YOUR_USERNAME/signals-dashboard.git
git push -u origin main
```

### 2. Add GitHub Secrets

Go to repo → Settings → Secrets and variables → Actions → New secret:

| Secret name         | Value                          |
|---------------------|--------------------------------|
| `CRYPTOQUANT_KEY`   | Your CryptoQuant API key       |
| `FRED_API_KEY`      | Your FRED API key              |
| `TELEGRAM_BOT_TOKEN`| Your Telegram bot token        |
| `TELEGRAM_CHAT_ID`  | Your Telegram chat/group ID    |
| `COINGECKO_API_KEY` | Your CoinGecko API key (optional) |

### 3. Trigger manually (first run)

Go to repo → Actions → "Fetch BTC Signals" → Run workflow.

After it completes, `public/data.json` will be populated with 365 days of signal data.

### 4. Deploy to Vercel

Connect your GitHub repo to Vercel. Set framework to "Other" (or Vite/CRA once you add the React app). Vercel auto-deploys on every push — including when GitHub Actions commits the updated `data.json`.

## Local testing

```bash
pip install -r requirements.txt
export CRYPTOQUANT_KEY="your_key"
export FRED_API_KEY="your_key"
python scripts/fetch_signals.py
# → writes public/data.json
```

## Signals

31 signals across 4 categories:

- **Valuation** (8): MVRV, SOPR, SOPR_Adj, STH_SOPR, LTH_SOPR, NVT, NVM, S2F_Dev
- **On-chain** (7): Exch_Reserve, Exch_Netflow, Whale_Ratio, MPI, Puell, SOPR_Ratio, Dormancy
- **Derivatives** (5): Lev_Ratio, SSR, Open_Interest, Coinbase_Prem, NRPL
- **Proxy/Macro** (11): MVRV_Proxy, Puell_Proxy, RealVol_30, RealVol_90, LR_1Y, LR_2Y_Z, VIX, DXY, SP500_Trend, Gold_90d, HY_Spread

Ternary scoring: rolling 730-day 80/20 percentile windows → `+1` (bearish), `0` (neutral), `-1` (bullish).

## Alert thresholds

Edit constants at the top of `scripts/fetch_signals.py`:

```python
ALERT_COMPOSITE_BEARISH = 0.5     # composite >= 0.5 → bearish alert
ALERT_COMPOSITE_BULLISH = -0.5    # composite <= -0.5 → bullish alert
ALERT_BTC_CRASH_PCT     = -10.0   # BTC 7-day return <= -10% → crash alert
ALERT_BTC_SURGE_PCT     = 15.0    # BTC 7-day return >= +15% → surge alert
```

## Cost

| Component      | Cost  |
|----------------|-------|
| GitHub Actions | Free (uses ~2 min/day of 2,000 free min/month) |
| Vercel         | Free  |
| Telegram Bot   | Free  |
| Signal APIs    | What you already pay |
| **Total new**  | **$0** |
