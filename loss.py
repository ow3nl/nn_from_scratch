"""Some common loss functions and their derivatives."""

import numpy as np
import math
from abc import ABC, abstractmethod


class Loss(ABC):
    """Abstract Base Class for loss functions."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def loss(
        self,
        data: np.ndarray[np.float64],
        target: np.ndarray[np.float64]
    ) -> np.float64:
        pass

    @abstractmethod
    def error(
        self,
        data: np.ndarray[np.float64],
        target: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        pass


class MSE(Loss):
    """Mean Squared Error loss."""
    
    def __init__(self) -> None:
        pass

    def loss(
        self,
        data: np.ndarray[np.float64], 
        target: np.ndarray[np.float64]
    ) -> np.float64:
        return np.average(np.square(data - target))

    def error(
        self,
        data: np.ndarray[np.float64],
        target: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return target - data


class NLL(Loss):
    """Negative Log-Likelihood loss, to be used with a softmax output layer."""
    
    def __init__(self) -> None:
        pass

    def loss(
        self,
        data: np.ndarray[np.float64], 
        target: np.ndarray[np.float64]
    ) -> np.float64:
        return -math.log(data[np.argmax(target)])
    
    def error(
        self,
        data: np.ndarray[np.float64],
        target: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return target - data


if __name__ == "__main__":
    data = np.array([0.4, 0.2, 0.1, 0.3])
    target = np.array([0, 1, 0, 0])

    M = MSE()
    L = NLL()
    print(M.loss(data, target))
    print(L.loss(data, target))


