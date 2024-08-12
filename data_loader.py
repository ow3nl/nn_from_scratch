"""Library to load MNIST image data."""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def load_data(dataset_name: str = "mnist", log: bool = False):
    """Load data. `dataset_name` must be \"mnist\" or \"fashion_mnist\"."""
    prefix = "my_datasets/" + dataset_name
    trn_img_bytes = Path(prefix + '/train-images-idx3-ubyte').read_bytes()
    trn_lbl_bytes = Path(prefix + '/train-labels-idx1-ubyte').read_bytes()
    tst_img_bytes = Path(prefix + '/t10k-images-idx3-ubyte').read_bytes()
    tst_lbl_bytes = Path(prefix + '/t10k-labels-idx1-ubyte').read_bytes()

    if log: print("Loading training images...")
    bc = 0  # byte counter
    
    magic_number = int.from_bytes(trn_img_bytes[bc:bc+4])
    bc += 4
    
    n_images = int.from_bytes(trn_img_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of images:", n_images)
    
    n_rows = int.from_bytes(trn_img_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of rows:", n_rows)
    
    n_cols = int.from_bytes(trn_img_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of columns:", n_cols)

    train_image_data = np.zeros((n_images, n_rows, n_cols), dtype=np.float64)
    for i in range(n_images):
        for j in range(n_rows):
            for k in range(n_cols):
                pixel = int.from_bytes(trn_img_bytes[bc:bc+1])
                train_image_data[i, j, k] = pixel / 256
                bc += 1
    
    if log: print("Loading training labels...")
    bc = 0

    magic_number = int.from_bytes(trn_lbl_bytes[bc:bc+4])
    bc += 4

    n_labels = int.from_bytes(trn_lbl_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of labels:", n_labels)

    train_label_data = np.zeros((n_images, 10), dtype=np.float64)
    for i in range(n_labels):
        pixel = int.from_bytes(trn_lbl_bytes[bc:bc+1])
        train_label_data[i, pixel] = 1
        if i < 10 and log:
            print(pixel)
            print(train_label_data[:10])
        bc += 1

    if log: print("Loading testing images...")
    bc = 0
    
    magic_number = int.from_bytes(tst_img_bytes[bc:bc+4])
    bc += 4
    
    n_images = int.from_bytes(tst_img_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of images:", n_images)
    
    n_rows = int.from_bytes(tst_img_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of rows:", n_rows)
    
    n_cols = int.from_bytes(tst_img_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of columns:", n_cols)

    test_image_data = np.zeros((n_images, n_rows, n_cols), dtype=np.float64)
    for i in range(n_images):
        for j in range(n_rows):
            for k in range(n_cols):
                pixel = int.from_bytes(tst_img_bytes[bc:bc+1])
                test_image_data[i, j, k] = pixel / 256
                bc += 1

    if log: print("Loading testing labels...")
    bc = 0

    magic_number = int.from_bytes(tst_lbl_bytes[bc:bc+4])
    bc += 4

    n_labels = int.from_bytes(tst_lbl_bytes[bc:bc+4])
    bc += 4
    if log: print("Number of labels:", n_labels)

    test_label_data = np.zeros((n_images), dtype=np.uint8)
    for i in range(n_labels):
        pixel = int.from_bytes(tst_lbl_bytes[bc:bc+1])
        test_label_data[i] = pixel
        bc += 1

    if log:
        for i in range(5):
            plt.imshow(train_image_data[i], interpolation="nearest",
                       cmap='gray')
            plt.title(str(np.argmax(train_label_data[i])))
            plt.show()
        for i in range(5):
            plt.imshow(test_image_data[i], interpolation="nearest", cmap='gray')
            plt.title(str(test_label_data[i]))
            plt.show()

    return train_image_data, train_label_data, test_image_data, test_label_data


if __name__ == "__main__":
    load_data(True)
