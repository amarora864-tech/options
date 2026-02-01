# Covered Call Strike Agent

This repo provides a small agent-style CLI that screens covered call strikes for tickers such as QQQ, SPY, INTC, and BBIO. The tool surfaces:

- VIX curve shape (9D, 1M, 3M, 6M, 1Y proxies).
- Implied volatility (IV) for each candidate call.
- IV / realized volatility (RV) ratio.
- Black-Scholes delta and gamma for each call.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python covered_call_agent.py --tickers QQQ SPY INTC BBIO --target-delta 0.2 --dte 35
```

## Notes

- Data is sourced via Yahoo Finance through `yfinance`.
- The risk-free rate uses the 13-week T-bill proxy (`^IRX`).
- The tool aims to pick the expiration closest to your target DTE and returns strikes near the target delta.
