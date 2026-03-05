# BTC Signals Dashboard

A daily BTC market signals dashboard that aggregates 31 on-chain, derivatives, valuation, and macro indicators into a single composite score. Data is fetched automatically every day and served as a static web app.

## How it works

A Python pipeline runs on a daily schedule, pulls data from multiple sources, computes ternary signal scores, and writes the results to `public/data.json`. The web app reads this file and renders charts and signal tables — no backend required.

## Signals

31 signals across 5 categories:

### Valuation (8)
| Signal | Description |
|---|---|
| MVRV | Market Value to Realized Value. High = overvalued. |
| SOPR | Spent Output Profit Ratio. Above 1 = coins moving in profit. |
| SOPR Adjusted | SOPR excluding same-day transactions. |
| STH SOPR | Short-Term Holder SOPR. High = recent buyers in profit. |
| LTH SOPR | Long-Term Holder SOPR. High = long-term holders distributing. |
| NVT Ratio | Network Value to Transactions. High = overvalued vs on-chain activity. |
| NVM Ratio | Network Value to Metcalfe. High = overvalued vs network growth. |
| S2F Deviation | Stock-to-Flow model deviation. High = price above model. |

### On-Chain (7)
| Signal | Description |
|---|---|
| Exchange Reserve | BTC held on Binance. Rising = potential sell pressure. |
| Exchange Netflow | Net BTC inflow to exchanges. Positive = sell pressure. |
| Whale Ratio | Top-10 exchange inflows / total inflows. High = whale distribution. |
| MPI | Miners' Position Index. High = miners selling. |
| Puell Multiple | Daily miner revenue / 365-day MA. High = miners overselling. |
| SOPR Ratio | LTH-SOPR / STH-SOPR ratio. |
| Dormancy Flow | Average dormancy of spent outputs. High = old coins moving. |

### Derivatives (5)
| Signal | Description |
|---|---|
| Leverage Ratio | Estimated leverage ratio on Binance. High = fragile market. |
| SSR | Stablecoin Supply Ratio. High = less stablecoin buying power. |
| Open Interest | Futures open interest on Binance. High = overheated market. |
| Coinbase Premium | BTC premium on Coinbase vs Binance. Positive = US buying pressure (bullish). |
| NRPL | Net Realized Profit/Loss. Positive = profit-taking. |

### Proxy / Computed (6)
| Signal | Description |
|---|---|
| Mayer Multiple | Price / 200-day MA. Free proxy for MVRV. |
| Puell Proxy | 14d price change / 365d price change. Miner revenue proxy. |
| Realized Vol 30d | 30-day annualized realized volatility. |
| Realized Vol 90d | 90-day annualized realized volatility. |
| 1Y Log Return | BTC 1-year log return. High = overbought momentum. |
| 2Y Z-Score | BTC 2-year price z-score. High = extended above long-run mean. |

### Macro (5)
| Signal | Description |
|---|---|
| VIX | CBOE Volatility Index. Spike = fear / contrarian bullish. |
| DXY | US Dollar Index. Strong USD = risk-off = bearish for BTC. |
| S&P 500 Trend | S&P 500 % above/below 200-day MA. Positive = risk-on = bullish for BTC. |
| Gold 90d Return | Rising gold = risk-off = bearish for BTC. |
| HY Credit Spread | High-yield credit spread. High = credit stress = bearish. |

## Scoring methodology

Each signal is scored on a ternary scale using a rolling 730-day (2-year) percentile window:

- `+1` (bearish) — signal above the 80th percentile
- `0` (neutral) — signal between 20th and 80th percentile
- `-1` (bullish) — signal below the 20th percentile

For inverse-direction signals (VIX, Coinbase Premium, SP500 Trend), the sign is flipped so that the final score is always intuitive: positive = bearish pressure, negative = bullish pressure.

The **composite score** is the simple mean of all available ternary scores on a given day, ranging from `-1` (fully bullish) to `+1` (fully bearish).

## Alerts

Telegram alerts are sent when any of the following thresholds are crossed:

| Condition | Threshold |
|---|---|
| Composite >= | `+0.5` → bearish alert |
| Composite <= | `-0.5` → bullish alert |
| BTC 7-day return <= | `-10%` → crash warning |
| BTC 7-day return >= | `+15%` → surge alert |

Thresholds can be adjusted at the top of `scripts/fetch_signals.py`.

## Data sources

| Source | Signals |
|---|---|
| CryptoQuant API | 20 on-chain and derivatives signals |
| FRED (St. Louis Fed) | HY Credit Spread |
| Yahoo Finance (yfinance) | BTC price, VIX, DXY, S&P 500, Gold |
| Computed | Mayer Multiple, Puell Proxy, Realized Vol, Log Returns, Z-Score |
