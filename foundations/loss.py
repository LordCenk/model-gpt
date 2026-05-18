
import numpy as np

class Solution:

    def binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-15
        loss = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true, y_pred):
        eps = 1e-15
        loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
        return round(loss, 4)