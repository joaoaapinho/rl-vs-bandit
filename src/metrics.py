"""Metrics - Financial performance metrics."""

import numpy as np


def sharpe_ratio(rewards):
    """Annualised risk-adjusted return (mean / std)."""
    return np.mean(rewards) / np.std(rewards)


def cumulative_pnl(rewards):
    """Running total of rewards over time."""
    return np.cumsum(rewards)


def max_drawdown(pnl):
    """Largest peak-to-trough decline in the PnL curve."""
    peak = np.maximum.accumulate(pnl)
    drawdown = peak - pnl
    return np.max(drawdown)
