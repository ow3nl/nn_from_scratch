# Neural Networks From Scratch

## About

This repository includes code for a neural network library built from scratch in Python and NumPy, and several notebooks which utilize this library to explore neural networks.

## Files

The standard python files (`.py` file extension) are library code. There may be minimal unit tests still included at the bottom, under the `if __name__ == "__main__"` statement.

The Jupyter notebooks (`.ipynb` file extension) are notebooks which I used the custom library to play around with some neural networks.
The notebooks are somewhat chronological, with later files sometimes building off of previous ones.
Some of the notebooks demonstrate creating standard well-known networks, while others explore new ideas I had.
More detailed descriptions of the contents of each notebook can be found within each file.
All files train and evaluate the networks on the MNIST and Fashion MNIST datasets.

The standard networks are in the following files:
- `basic_network.ipynb`
- `improved_network.ipynb`
- `deep_network.ipynb`
- `convolutional_network.ipynb`

The experimental networks are in the following files, in no particular order:
- `broad_updates.ipynb`
- `feature_preprocessing.ipynb`
- `layer_freezing.ipynb`
- `weight_initialization.ipynb`

The experimental networks explored in `feature_preprocessing.ipynb` and `layer_freezing.ipynb` showed better performance than the standard network.
