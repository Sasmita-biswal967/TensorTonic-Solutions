def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    # Write code here
    n = len(rewards)
    advantages = [0.0] * n

    next_adv = 0.0

    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        next_adv = delta + gamma * lam * next_adv
        advantages[t] = next_adv

    return advantages
    pass