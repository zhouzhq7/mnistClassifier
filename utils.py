import numpy as np
from array import array
import struct
from scipy import misc
import matplotlib.pyplot as plt

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
            tmp = np.reshape(tmp, [rows, cols])
            images.append(tmp)

        return np.array(images)

a = load_label_file('./data/t10k-labels.idx1-ubyte')
b = load_image_file('./data/t10k-images.idx3-ubyte')
img = b[0,:,:]
plt.imshow(img)
plt.show()
pass

