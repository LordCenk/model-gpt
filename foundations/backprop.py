import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        # Forward pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        # Common term from chain rule
        dz = (y_hat - y_true) * y_hat * (1 - y_hat)

        # Gradients
        dL_dw = dz * x
        dL_db = dz

        # Round to 5 decimals
        return np.round(dL_dw, 5), round(float(dL_db), 5)