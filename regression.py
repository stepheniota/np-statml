""" Regression models. """
import numpy as np

# local helper function
from helper import prepare_data

class LinearRegression:
    """ Ordinary least squares Linear Regression. """
    def __init__(self, lr=1e-4, n_epoch=1000, fit_intercept=True):
        self.lr = lr
        self.n_epoch = n_epoch
        self.weights = None
        self.fit_intercept = None


    def fit(self, X, y, sample_weights=None):
        """ Fit linear model using GD. """
        X, y = prepare_data(X, y)
        _, D = X.shape

        if sample_weights:
            assert sample_weights.shape == ((D, 1))
            self.weights = sample_weights
        else:
            self.weights = np.zeros((D, 1))

        for _ in range(self.n_epoch):
            y_hat = self.predict(X)
            delta_w = X.T @ y_hat
            delta_w = delta_w - (X.T @ y)
            self.weights -= self.lr * delta_w


    def fit_closed_form(self, X, y):
        """ Fit linear model using closed form solution.
            Computes MLE estimate of argmin NLL(w).

            :math: W^* = (X^T @ X)^{-1} @ X^T @ y

            ref ~ Murphy MLaPP [Ch.7.3; pg.222]
        """
        X, y = prepare_data(X, y)
        self.weights = np.invert(X.T @ X) @ X.T @ y


    def predict(self, X):
        """ Predict linear model. """
        X = prepare_data(X)
        return X @ self.weights


    def loss(self, X, y):
        """ Score model performance using Mean Squared Error
            :math: mse = \frac{1}{N} \sum_{i=1}^{N} (y_hat - y)^2
        """
        X, y = prepare_data(X, y)
        y_hat = self.predict(X)
        return np.square(y_hat - y).mean()


class LogisticRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def loss(self, X, y):
        pass
