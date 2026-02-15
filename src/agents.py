"""Agents - Trading strategy implementations (bandit vs Q-learning)."""

import numpy as np


def bandit_strategy(prices):
    """Momentum bandit: action = sign of last return, maximises immediate reward."""
    returns = np.diff(prices)
    rewards = np.zeros(len(returns))
    for t in range(1, len(returns)):
        action = np.sign(returns[t - 1])  # Buy if last move was up, sell if down
        rewards[t] = action * returns[t]  # PnL = position * price change
    return rewards


def q_learning(prices, alpha=0.1, gamma=0.95, epsilon=0.1):
    """Tabular Q-learning agent. State = discretised last return, action = {-1, +1}.

    Because prices follow a random walk (actions don't alter state transitions),
    the Bellman update collapses to immediate-reward maximisation - same as bandit.
    """
    returns = np.diff(prices)
    Q, actions = {}, [-1, 1]  # Q-table and action space
    rewards = np.zeros(len(returns))

    def state(r):
        """Discretise return into {-1, 0, +1} (down / flat / up)."""
        return int(np.sign(r)) if abs(r) > 0.5 else 0

    for t in range(1, len(returns)):
        s = state(returns[t - 1])

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            a = np.random.choice(actions)
        else:
            a = actions[np.argmax([Q.get((s, a), 0.0) for a in actions])]

        rewards[t] = a * returns[t]
        ns = state(returns[t])

        # TD(0) Bellman update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
        Q[(s, a)] = Q.get((s, a), 0.0) + alpha * (
            rewards[t] + gamma * max(Q.get((ns, a2), 0.0) for a2 in actions) - Q.get((s, a), 0.0)
        )
    return rewards
