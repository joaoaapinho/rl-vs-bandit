"""Environment - Market simulation for price-taker world."""

import numpy as np


def simulate_price_taker(T=10000, mu=0.0, sigma=0.01, S0=100.0):
    """Geometric Brownian Motion: log-returns are normal, prices are log-normal."""
    log_returns = (mu - 0.5 * sigma**2) + sigma * np.random.normal(0, 1, T)
    prices = S0 * np.exp(np.cumsum(log_returns))
    return prices
