""" Implementation of different distance functions. """
import numpy as np


class Distances:
    @staticmethod
    def canberra_distance(point1, point2):
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        # allow division by zero, then set all inf/nan to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.true_divide(np.abs(point1 - point2), np.abs(point1) + np.abs(point2))
            d[d == np.inf] = 0
            np.nan_to_num(d, copy=False)

        return np.sum(d)


    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1). Here we take p=3.
        Ref - https://en.wikipedia.org/wiki/Minkowski_distance
        """
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        return np.power(np.sum(np.power(np.abs(point1 - point2), 3)), 1 / 3)

    @staticmethod
    def euclidean_distance(point1, point2):
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        return np.linalg.norm(point1 - point2)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        return np.dot(point1, point2)

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        return (1 - (np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))))

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        if not isinstance(point1, np.ndarray):
            point1 = np.array(point1)
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)

        return -np.exp(-0.5 * np.dot(point1 - point2, point1 - point2))

    