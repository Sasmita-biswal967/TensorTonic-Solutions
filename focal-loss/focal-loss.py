import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Clip probabilities to avoid log(0)
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Apply focal loss elementwise
    loss = -((1 - p) ** gamma * y * np.log(p) +
             p ** gamma * (1 - y) * np.log(1 - p))

    return float(np.mean(loss))