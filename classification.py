""" Discriminitive classifier methods. """

import numpy as np
# local imports
from helper import to_numpy, prepare_data


class LogisticRegression:
    def __init__(self, lr=1e-4, n_epoch=1000, fit_intercept=True):
        """ Logisitic regression binary classifier. """
        self.lr = lr
        self.n_epoch = n_epoch
        self.fit_intercept = fit_intercept
        self.weights = None
        self.threshold = 0.5


    def fit(self, X, y):
        """ Fit model using vanilla gradient descent. """
        X, y = prepare_data(X, y)

        _, D = X.shape
        self.weights = np.zeros((D, 1))

        for _ in range(self.n_epoch):
            y_hat = self.predict(X)
            delta_w = self.sigmoid(y_hat) - y_hat
            delta_w = np.einsum('ij,jk->ik', X, delta_w)
            self.weights -= self.lr * delta_w


    def predict(self, X):
        """ Predict some label y {0, 1} for each input x_i. """
        z = np.einsum('ij,jk->ik', X, self.weights)  # Doesn't look as nice as `@`
        y_hat = self.sigmoid(z)                      # but I wanted to try it lol
        return int(y_hat >= self.threshold)


    def loss(self, X, y):
        """ Cross entropy loss. """
        z = np.einsum('ij,jk->ik', X, self.weights)
        y_hat = self.sigmoid(z)
        loss = y * np.log2(y_hat) + (1 - y) * np.log2(1 - y_hat)
        return - loss.sum()


    def sigmoid(self, z):
        """ Numerically stable sigmoid function.

        Ref ~ Tim Viera [exp-normalize-trick]
        (http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)

        Another potential solution: `return np.exp(-np.logaddexp(0, -z))`
        Not looked into yet, see ~ [Neil G](https://stackoverflow.com/a/29863846)
        """
        if z >= 0:
            x = np.exp(-z)
            return 1 / (1 + x)
        else:
            # z < 0 -> z is very small
            # this avoids zero in the denominator
            x = np.exp(z)
            return x / (1 + x)
