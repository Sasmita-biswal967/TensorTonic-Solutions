import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here\
    z1=np.array(Z1, dtype=float)
    z2=np.array(Z2, dtype=float)

    l=(z1@z2.T)/temperature
    l=l-np.max(l,axis=1,keepdims=True)

    exp_l=np.exp(l)
    pos=np.diag(exp_l)
    row_sum=np.sum(exp_l,axis=1)
    loss = -np.log(pos / row_sum)

    return float(np.mean(loss))
    
    pass