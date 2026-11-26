import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    # Write code here
    returns = np.zeros(len(rewards))
    G = 0

    # Compute discounted returns backward
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Subtract the value baseline of each visited state
    advantages = returns - np.array([V[s] for s in states])

    return np.round(advantages, 4)
    pass