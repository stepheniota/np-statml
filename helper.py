""" Helper functions when dealing with different input types. """
import numpy as np

def prepare_data(X, y=None):
    X = to_numpy(X)
    if len(X) == 1:
        X = np.expand_dims(X, axis=0)
    if y:
        y = to_numpy(y)
        N, _ = X.shape
        if y.shape == (N,):
            y = np.expand_dims(y, axis=0)
        return X, y
    return X

def to_numpy(X):
    if not isinstance(X, np.ndarray):
        return np.array(X)
    return X