""" Regression models """
import numpy as np

class LinearRegression:
    """ Ordinary least squares Linear Regression. """
    def __init__(self, lr=0.001, n_epoch=1000):
        self.lr = lr
        self.n_epoch = n_epoch
        self.weights = None
        self.fit_intercept = None
        self.seed = None
        self.rng = np.random.default_rng(seed=self.seed)


    def fit(self, X, y):
        """ Fit linear model using GD. """
        X, y = self._prepare_data(X, y)
        N = len(X)

        self.weights = np.zeros((D, 1))
        for _ in range(self.n_epoch):
            y_hat = self.predict(X)
            delta_w = X.T @ y_hat
            delta_w = delta_w - (X.T @ y)
            self.weights -= self.lr * delta_w


    def fit_closed_form(self, X, y, sample_weight=None):
        """ Fit linear model using closed form solution.
            Computes MLE estimate of argmin NLL(w).

            :math: W^* = (X^T @ X)^{-1} @ X^T @ y

            ref ~ Murphy MLaPP [Ch.7.3; pg.222]
        """
        X, y = self._prepare_data(X, y)
        self.weights = np.invert(X.T @ X) @ X.T @ y


    def predict(self, X):
        """ Predict linear model. """
        X, _ = self._prepare_data(X)
        return X @ self.weights


    def mse(self, X, y):
        """ Score model performance using Mean Squared Error
            :math: mse = \frac{1}{N} \sum_{i=1}^{N} (y_hat - y)^2
        """
        X, y = self._prepare_data(X, y)

        y_hat = self.predict(X)
        return np.square(y_hat - y).mean()


    def _prepare_data(self, X, y=None):
        X = np.array(X)
        if len(X) == 1:
            X = np.expand_dims(X, axis=0)
        y = np.array(y)
        N, D = X.shape
        if y.shape == (N,):
            y = np.expand_dims(y, axis=1)
        return X, y


if __name__ == '__main__':
    print('Some manual tests.')
    N, D = 100, 3
    X = np.random.random(size=(N, D))
    X += np.random.normal(0, 2, size=(N, D))
    y = np.dot(X, [1,1,1])
    lr = LinearRegression()
    lr.fit(X, y)
    x_test = np.array(([[1, 1, 1], [2,2,2]]))
    y_hat = lr.predict(x_test)
    print(y_hat)
    print(lr.mse(X, y))