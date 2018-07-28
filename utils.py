import numpy as np
from array import array
import struct

def load_label_file(filename):
    with open(filename, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError('MSB is mismatched.')

        labels = np.array(array('B', f.read()))

    return labels

def load_image_file(filename):
    with open(filename, 'rb') as f:
        magic, size, rows, cols = struct.unpack('>IIII', f.read(16))

        if magic != 2051:
            raise  ValueError('MSB is mismatched.')

        images_byte_data = np.array(array("B", f.read()))

        image_size = rows*cols

        images = []

        for i in range(size):
            tmp = images_byte_data[i*image_size:(i+1)*image_size]
            images.append(tmp)

        return np.array(images)


def load_mnist_from_files(x_train_filename, y_train_filename, x_val_filename, y_val_filename):
    X, y = {}, {}
    X['train'] = load_image_file(x_train_filename)
    y['train'] = load_label_file(y_train_filename)
    X['val'] = load_image_file(x_val_filename)
    y['val'] = load_label_file(y_val_filename)

    return X, y

def load_mnist():
    return load_mnist_from_files(
        './data/train-images.idx3-ubyte',
        './data/train-labels.idx1-ubyte',
        './data/t10k-images.idx3-ubyte',
        './data/t10k-labels.idx1-ubyte'
    )
