import numpy as np
## local import
from distances import Distances

DISTANCE_FUNCS = {
        'canberra': Distances.canberra_distance,
        'minkowski': Distances.minkowski_distance,
        'euclidean': Distances.euclidean_distance,
        'gaussian': Distances.gaussian_kernel_distance,
        'inner_prod': Distances.inner_product_distance,
        'cosine_dist': Distances.cosine_similarity_distance,
    }

class KNN:
    """ K Nearest Neighbors Classifier. """
    def __init__(self, k_neighbors, distance_func):
        self.k = k_neighbors
        self.distance = DISTANCE_FUNCS[distance_func]

    def fit(self, X, y):
        """ Fit the kNN classifier from the training dataset. """
        self.features, self.labels = X, y

    def predict(self, X):
        """ Predict the class labels for the provided data. """
        predicted_labels = np.zeros(len(X))

        for i, point in enumerate(X):
            nei = self.get_k_neighbors(point)
            predicted_labels[i] = np.bincount(nei).argmax()

        return predicted_labels

    def kneighbors(self, x):
        """ Find the K-neighbors of a point. """
        nei = np.empty((len(self.X)), 
                       dtype=np.dtype([('d', np.float64), ('y', np.int64)]))

        for i, point in enumerate(self.X):
            nei[i] = (self.distance_function(point, x), self.y_train[i])

        # sorts increaseing order according to distance (first arg)
        nei.sort(order='d')

        return np.array([x[1] for x in nei[:self.k]])
 