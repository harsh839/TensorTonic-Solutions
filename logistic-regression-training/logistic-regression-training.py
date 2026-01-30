import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    N, D = X.shape

   
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(steps):
       
        z = X @ w + b

        
        y_hat = _sigmoid(z)

        
        error = y_hat - y
        dw = (1 / N) * (X.T @ error)
        db = (1 / N) * np.sum(error)

        
        w -= lr * dw
        b -= lr * db

    return w, b



    