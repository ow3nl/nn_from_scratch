"""Implementation of a vanilla neural network"""

from typing import *
from layers import *
from loss import *
import random


class Network:

    def __init__(
            self,
            layers: List[Layer | TrainableLayer],
            learning_rate: np.float64,
            mini_batch_size: int
    ) -> None:
        """Initialize the network."""
        self.layers = layers
        self.input_size = self.layers[0].input_size
        self.output_size = self.layers[-1].output_size
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
    
    def reset(self) -> None:
        """Re-initialize all parameters of the network."""
        for layer in self.layers:
            if layer.trainable:
                layer.reset()
    
    def save(self, filepath: str) -> None:
        """Save the current state of the network."""
        pass  # TODO
    
    def set_training(self, value: bool) -> None:
        for layer in self.layers:
            layer.training = value
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Feed a given input forward and return the ouptut.

        This method can be overwritten for non sequential networks.
        """
        data = input_data.copy()
        for layer in self.layers:
            data = layer.forward(data)

        return data
    
    def iter_batches(
            self,
            training_data: np.ndarray
        ) -> Iterator[np.ndarray]:
        """Iterator over the training data in mini-batches."""
        for i in range(0, len(training_data), self.mini_batch_size):
            yield training_data[i:i+self.mini_batch_size]

    def train(
        self,
        epochs: int,
        training_data: np.ndarray,
        loss: Loss,
        test_data: Optional[np.ndarray] = None,
        suppress: bool = False,
        frozen_layers: Optional[List[int]] = None,
    ) -> None:
        """Train the network on a set of training data."""
        for e in range(epochs):
            random.shuffle(training_data)

            if frozen_layers is None:
                self.set_training(True)
            else:
                for i, layer in enumerate(self.layers):
                    if layer.trainable:
                        if i not in frozen_layers:
                            layer.training = True
                        else:
                            layer.training = False
            
            for batch in self.iter_batches(training_data):
                self.train_batch(loss, batch, len(training_data))
            
            if test_data is not None:
                self.set_training(False)
                correct = self.evaluate(test_data)
                percent = round(correct / len(test_data) * 100, 1)
                if not suppress:
                    print(f"Epoch {e+1}: {correct} / {len(test_data)} ({percent}%)")
            else:
                if not suppress:
                    print(f"Epoch {e+1} finished")
    
    def train_batch(
        self,
        loss: Loss,
        mini_batch: np.ndarray,
        training_size: int
    ) -> None:
        """Train the network on a single mini-batch."""
        for input_data, target_data in mini_batch:
            output_data = self.forward(input_data)

            error = loss.error(output_data, target_data)

            for layer in reversed(self.layers):
                error = layer.backprop(error)

        for layer in self.layers:
            if layer.trainable:
                layer.regularize_parameters(training_size, self.learning_rate)
                layer.update_parameters(self.learning_rate / len(mini_batch))
    
    def evaluate(self, test_data: np.ndarray, data: bool = False) -> int:
        """Evaluate model performance on given test_data."""
        if not data:
            return sum((int(np.argmax(self.forward(x)) == y) for x, y in test_data))
        
        correct = 0
        misses = [0 for _ in range(self.output_size)]
        for x, y in test_data:
            if np.argmax(self.forward(x)) == y:
                correct += 1
            else:
                misses[y] += 1
        
        print(misses)
        
        return correct
