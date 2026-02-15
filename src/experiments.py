"""Experiments - Monte Carlo hypothesis testing for bandit collapse."""

import numpy as np
from scipy import stats
from environment import simulate_price_taker
from agents import bandit_strategy, q_learning
from metrics import sharpe_ratio


def run_trials(n_trials=100, T=10000):
    """Run n independent simulations, return Sharpe(RL) - Sharpe(Bandit) per trial."""
    diffs = []
    for _ in range(n_trials):
        prices = simulate_price_taker(T)
        sr_bandit = sharpe_ratio(bandit_strategy(prices))
        sr_rl = sharpe_ratio(q_learning(prices))
        diffs.append(sr_rl - sr_bandit)
    return np.array(diffs)


def test_hypothesis(diffs):
    """One-sample t-test: H₀ is mean Sharpe difference = 0 (no RL advantage)."""
    return stats.ttest_1samp(diffs, 0)
