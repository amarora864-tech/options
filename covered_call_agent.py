#!/usr/bin/env python3
"""Covered call screening agent for strike selection and volatility context."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

VIX_TICKERS = {
    "VIX9D": "^VIX9D",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VIX1Y": "^VIX1Y",
}


@dataclass
class CallCandidate:
    ticker: str
    expiration: str
    dte: int
    strike: float
    last_price: float
    iv: float
    iv_rv: float
    delta: float
    gamma: float


@dataclass
class VixPoint:
    label: str
    value: float


def get_risk_free_rate() -> float:
    """Fetch a proxy for the risk-free rate using the 13-week T-bill (^IRX)."""
    irx = yf.Ticker("^IRX").history(period="10d")
    if irx.empty:
        return 0.02
    return float(irx["Close"].dropna().iloc[-1]) / 100.0


def realized_volatility(prices: pd.Series, trading_days: int = 252) -> float:
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        return float("nan")
    return float(returns.std() * np.sqrt(trading_days))


def black_scholes_delta_gamma(spot: float, strike: float, t: float, r: float, iv: float) -> tuple[float, float]:
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return float("nan"), float("nan")
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (spot * iv * np.sqrt(t)))
    return delta, gamma


def vix_curve() -> List[VixPoint]:
    points: List[VixPoint] = []
    for label, ticker in VIX_TICKERS.items():
        history = yf.Ticker(ticker).history(period="5d")
        if history.empty:
            continue
        value = float(history["Close"].dropna().iloc[-1])
        points.append(VixPoint(label=label, value=value))
    return points


def pick_expiration(expirations: Iterable[str], target_dte: int) -> Optional[str]:
    today = datetime.utcnow().date()
    best_exp = None
    best_diff = None
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_dte)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_exp = exp
    return best_exp


def screen_calls(
    ticker: str,
    target_delta: float,
    dte_target: int,
    delta_band: float,
    max_rows: int,
) -> List[CallCandidate]:
    yf_ticker = yf.Ticker(ticker)
    history = yf_ticker.history(period="1y")
    if history.empty:
        return []

    spot = float(history["Close"].dropna().iloc[-1])
    rv = realized_volatility(history["Close"].dropna())

    expirations = yf_ticker.options
    if not expirations:
        return []

    expiration = pick_expiration(expirations, dte_target)
    if not expiration:
        return []

    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    dte = (exp_date - datetime.utcnow().date()).days
    t = dte / 365.0

    chain = yf_ticker.option_chain(expiration)
    calls = chain.calls.copy()
    if calls.empty:
        return []

    r = get_risk_free_rate()

    candidates: List[CallCandidate] = []
    for _, row in calls.iterrows():
        iv = float(row.get("impliedVolatility", np.nan))
        if not np.isfinite(iv) or iv <= 0:
            continue
        delta, gamma = black_scholes_delta_gamma(
            spot=spot,
            strike=float(row["strike"]),
            t=t,
            r=r,
            iv=iv,
        )
        if not np.isfinite(delta):
            continue
        if abs(delta - target_delta) > delta_band:
            continue
        iv_rv = iv / rv if rv and np.isfinite(rv) and rv > 0 else float("nan")
        candidates.append(
            CallCandidate(
                ticker=ticker,
                expiration=expiration,
                dte=dte,
                strike=float(row["strike"]),
                last_price=float(row.get("lastPrice", np.nan)),
                iv=iv,
                iv_rv=iv_rv,
                delta=delta,
                gamma=gamma,
            )
        )

    candidates.sort(key=lambda item: abs(item.delta - target_delta))
    return candidates[:max_rows]


def format_vix_curve(points: List[VixPoint]) -> str:
    if not points:
        return "VIX curve data unavailable."
    labels = [p.label for p in points]
    values = [p.value for p in points]
    curve_df = pd.DataFrame({"Label": labels, "VIX": values})
    curve_df["Change_vs_VIX"] = curve_df["VIX"] - curve_df.loc[curve_df["Label"] == "VIX", "VIX"].iloc[0]
    return curve_df.to_string(index=False)


def format_candidates(candidates: List[CallCandidate]) -> str:
    if not candidates:
        return "No candidates found."
    df = pd.DataFrame(
        [
            {
                "Ticker": c.ticker,
                "Expiration": c.expiration,
                "DTE": c.dte,
                "Strike": c.strike,
                "Last": c.last_price,
                "IV": c.iv,
                "IV/RV": c.iv_rv,
                "Delta": c.delta,
                "Gamma": c.gamma,
            }
            for c in candidates
        ]
    )
    return df.to_string(index=False, float_format=lambda x: f"{x:0.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen covered call strikes with VIX curve and IV/realized-vol context."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["QQQ", "SPY", "INTC", "BBIO"],
        help="Tickers to screen.",
    )
    parser.add_argument("--target-delta", type=float, default=0.2, help="Target call delta.")
    parser.add_argument("--delta-band", type=float, default=0.1, help="Allowed delta band.")
    parser.add_argument("--dte", type=int, default=35, help="Target days to expiration.")
    parser.add_argument("--max-rows", type=int, default=5, help="Max rows per ticker.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== VIX Curve ===")
    print(format_vix_curve(vix_curve()))
    print()

    for ticker in args.tickers:
        print(f"=== Covered Call Screen: {ticker} ===")
        candidates = screen_calls(
            ticker=ticker,
            target_delta=args.target_delta,
            dte_target=args.dte,
            delta_band=args.delta_band,
            max_rows=args.max_rows,
        )
        print(format_candidates(candidates))
        print()


if __name__ == "__main__":
    main()
