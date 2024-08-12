"""A module with an implementation of commonly used nerual network layers."""

from typing import *
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Layer(ABC):
    """
    Abstract base class for neural network layers.
    
    Each layer is treated as a function that takes an input array and returns an
    output array.
    """

    @abstractmethod
    def __init__(
        self,
        input_size: int | Tuple[int, ...],
        output_size: int | Tuple[int, ...],
        trainable: bool = False
    ) -> None:
        """Initialize the layer."""
        self.input_size = input_size
        self.output_size = output_size

        self.trainable = trainable

        self.training = True

    @abstractmethod
    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Forward pass. If `training` is set to `True`, stores input data."""
        pass
    
    @abstractmethod
    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Returns the error of the previous layer."""
        pass


class TrainableLayer(Layer):
    """
    Abstract base class for trainable neural network layers.
    
    In addition to standard layer methods, trainable layers have `reset` and
    `update_parameters`.
    """

    @abstractmethod
    def __init__(
        self,
        input_size: int | Tuple[int],
        output_size: int | Tuple[int],
        regularization_parameter: np.float64 = 0.
    ) -> None:
        super().__init__(input_size, output_size, True)
        self.regularization_parameter = regularization_parameter
        
        # If training, inputs will be stored for backpropagation.
        self.training = True
        self.input: np.ndarray | None = None

    @abstractmethod
    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Forward pass. If `training` is set to `True`, stores input data."""
        pass
    
    @abstractmethod
    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Returns the error of the previous layer."""
        pass
    
    @abstractmethod
    def regularize_parameters(
        self,
        training_size: int,
        learning_rate: np.float64
    ) -> None:
        """Regularize the trainable parameters of the layer."""
        pass

    @abstractmethod
    def update_parameters(self, learning_rate: np.float64, *args) -> None:
        """Update layer parameters according to currently stored gradients."""
        pass

    def reset(self) -> None:
        """Reset trainable parameters."""
        pass


class Sigmoid(Layer):
    """Implementation of a sigmoid layer."""

    def __init__(
        self,
        input_size: int | Tuple[int]
    ) -> None:
        super().__init__(input_size, input_size)
    
    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Apply the sigmoid function to every element in the input_data."""
        if self.training:
            self.input = input_data.copy()
        return 1 / (1 + np.exp(-input_data))

    def backprop(self, error: np.ndarray) -> np.ndarray:
        x = self.forward(self.input)
        return error * x * (1 - x)


class ReLU(Layer):
    """Implementation of a ReLU layer."""

    def __init__(
        self,
        input_size: int | Tuple[int]
    ) -> None:
        super().__init__(input_size, input_size)

    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Apply the ReLU function to every element in the input_data."""
        if self.training:
            self.input = input_data.copy()
        
        np.maximum(input_data, 0, input_data)
        return input_data

    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Backpropagate the error according to the derivative of the ReLU."""
        return error * (self.input > 0)


class LeakyReLU(Layer):
    """Implementation of a Leaky ReLU layer."""

    def __init__(
        self,
        input_size: int | Tuple[int],
        alpha: np.float64 = 0.2
    ) -> None:
        super().__init__(input_size, input_size)
        self.alpha = alpha

    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Apply the Leaky ReLU function to every element in the input_data."""
        if self.training:
            self.input = input_data.copy()

        np.maximum(input_data, self.alpha * input_data, input_data)
        return input_data

    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Backpropagate the error using the derivative of the Leaky ReLU."""
        error[self.input < 0] *= self.alpha
        return error


class Softmax(Layer):
    """Implementation of a softmax layer."""

    def __init__(
        self,
        input_size: int
    ) -> None:
        super().__init__(input_size, input_size)

    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Pass input layer through ReLU """
        ex = np.exp(input_data)
        return ex / ex.sum()

    def backprop(self, error: np.ndarray) -> np.ndarray:
        return error


class Flatten(Layer):
    
    def __init__(self, input_size: Tuple[int, ...]) -> None:
        out_size = 1
        for dim in input_size:
            out_size *= dim
        super().__init__(input_size, out_size)

    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Pass input layer through ReLU """
        return input_data.flatten()

    def backprop(self, error: np.ndarray) -> np.ndarray:
        return error.reshape(self.input_size)


class Dense(TrainableLayer):
    """Implementation of a dense (fully connected) layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        regularization_parameter: np.float64 = 0.,
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None
    ) -> None:
        super().__init__(input_size, output_size, regularization_parameter)
        
        if weights is None:
            self.weights = np.random.randn(self.output_size, self.input_size)
            # Scale weights to standardize output distribution
            self.weights /= np.sqrt(self.input_size)
        else:
            self.weights = weights

        if biases is None:
            self.biases = np.random.randn(self.output_size)
        else:
            self.biases = biases
        
        # nabla_w and nabla_b are derivatives of weights and biases, summed over
        # a mini batch.
        self.nabla_w = np.zeros(self.weights.shape, np.float64)
        self.nabla_b = np.zeros(self.biases.shape, np.float64)

    def reset(self) -> None:
        self.weights = np.random.randn(self.output_size, self.input_size)
        self.weights /= np.sqrt(self.input_size)
        self.biases = np.random.randn(self.output_size)
        self.nabla_w.fill(0.)
        self.nabla_b.fill(0.)

    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Pass some data forward. If training, stores input."""
        if self.training:
            self.input = input_data.copy()
        
        return self.biases + np.dot(self.weights, input_data)

    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Backpropagate error. If training, updates gradients."""
        if self.training:
            self.nabla_b -= error
            self.nabla_w -= np.outer(error, self.input)
        return np.dot(self.weights.transpose(), error)

    def regularize_parameters(
        self,
        training_size: int,
        learning_rate: np.float64
    ) -> None:
        """Regularize weights."""
        self.weights *= (1 - learning_rate * (
            self.regularization_parameter / training_size
        ))

    def update_parameters(
            self,
            learning_rate: np.float64,
    ) -> None:
        """Update layer parameters using currently stored gradients."""
        self.biases -= self.nabla_b * learning_rate
        self.weights -= self.nabla_w * learning_rate

        self.nabla_w.fill(0.)
        self.nabla_b.fill(0.)


class Conv2D(TrainableLayer):

    def __init__(
        self,
        input_size: Tuple[int, int],
        kernel_size: Tuple[int, int],
        zero_padding: int = 0,
        regularization_parameter: np.float64 = 0.,
        kernel: Optional[np.ndarray[np.float64]] = None,
        bias: Optional[np.ndarray[np.float64]] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.zero_padding = zero_padding
        
        output_size = (
            input_size[0] + 2*(zero_padding) - kernel_size[0] + 1,
            input_size[1] + 2*(zero_padding) - kernel_size[1] + 1,
        )
        super().__init__(input_size, output_size, regularization_parameter)

        if kernel is None:
            self.kernel = np.random.randn(*kernel_size)
            # Scale weights to standardize output distribution
            self.kernel /= np.sqrt(kernel_size[0] * kernel_size[1])
        else:
            self.kernel = kernel

        if bias is None:
            self.biases = np.random.randn(*self.output_size)
        else:
            self.biases = bias

        self.nabla_kernel = np.zeros(kernel_size)
        self.nabla_b = np.zeros(output_size)

    def forward(
        self,
        input_data: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        if self.training:
            self.input = input_data.copy()
        
        if self.zero_padding > 0:
            input_data = np.pad(
                input_data,
                self.zero_padding
            )
        
        w, h = self.kernel_size
        output_data = np.zeros(self.output_size)
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                output_data[i, j] = np.sum(
                    self.kernel * input_data[i:i+w, j:j+h]
                )
    
        return output_data + self.biases
    
    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Backpropagate error. If training, updates gradients."""
        w, h = self.kernel_size
        
        if self.training:
            self.nabla_b -= error

            for i in range(self.output_size[0]):
                for j in range(self.output_size[1]):
                    self.nabla_kernel -= (
                        error[i,j] * self.input[i:i+w, j:j+h]
                    )
        
        new_error = np.zeros(self.input_size)
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                new_error[i:i+w, j:j+h] += error[i, j] * self.kernel
        
        return new_error
    
    def regularize_parameters(
        self,
        training_size: int,
        learning_rate: np.float64
    ) -> None:
        """Regularize the kernel weights."""
        self.kernel *= (1 - learning_rate * (
            self.regularization_parameter / training_size
        ))

    def update_parameters(self, learning_rate: np.float64) -> None:
        """Update layer parameters according to currently stored gradients."""
        self.kernel -= self.nabla_kernel * learning_rate
        self.biases -= self.nabla_b * learning_rate

        self.nabla_kernel.fill(0.)
        self.nabla_b.fill(0.)

    def reset(self) -> None:
        """Reset trainable parameters."""
        self.kernel = np.random.randn(*self.kernel_size)
        self.kernel /= np.sqrt(self.kernel_size[0] * self.kernel_size[1])
        self.biases = np.random.randn(*self.output_size)

        self.nabla_kernel.fill(0.)
        self.nabla_b.fill(0.)


class MaxPool2D(Layer):
    """Implementation of a max pooling layer."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        pool_size: int
    ) -> None:
        self.pool_size = pool_size
        x, y = input_size
        # (a-1) / b + 1 is equal to ceil(a / b)
        output_size = ((x-1) // pool_size + 1, (y-1) // pool_size + 1)
        super().__init__(input_size, output_size, False)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Apply the ReLU function and max pooling on the input."""
        if self.training:
            self.input_data = input_data

        out = np.zeros(self.output_size, dtype=np.float64)
        for i in range(self.output_size[0]):
            x = i * self.pool_size
            for j in range(self.output_size[1]):
                y = j * self.pool_size
                out[i, j] = np.max(
                    input_data[x:x+self.pool_size, y:y+self.pool_size]
                )
        return out

    def backprop(self, error: np.ndarray) -> np.ndarray:
        """Backpropagate the error."""
        out = np.zeros(self.input_size, dtype=np.float64)
        for i in range(self.output_size[0]):
            x = i * self.pool_size
            for j in range(self.output_size[1]):
                y = j * self.pool_size
                max_idx = np.argmax(
                    self.input_data[x:x+self.pool_size, y:y+self.pool_size]
                )
                a, b = max_idx // self.pool_size, max_idx % self.pool_size
                out[x+a, y+b] += error[i, j]
        
        return out


if __name__ == "__main__":
    M = MaxPool2D((4, 4), 2)

    A = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [16, 15, 14, 13],
        [12, 11, 10, 9]
    ])

    print(M.forward(A))

    error = np.array([
        [10, -10],
        [100, -100]
    ])

    print(M.backprop(error))
