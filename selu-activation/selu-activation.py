import math

def selu(x):
    """
    Apply SELU activation element-wise to a list.

    Parameters:
        x : list of numbers

    Returns:
        list of floats (same length as x)
    """

    
    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    return [
        float(scale * val) if val > 0
        else float(scale * alpha * (math.exp(val) - 1))
        for val in x
    ]
