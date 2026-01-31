import math

def elu(x, alpha=1.0):
    """
    Apply ELU activation element-wise to a list.

    Parameters:
        x     : list of numbers
        alpha : non-negative float

    Returns:
        list of floats (same length as x)
    """
    return [
        float(val) if val > 0 else float(alpha * (math.exp(val) - 1))
        for val in x
    ]
