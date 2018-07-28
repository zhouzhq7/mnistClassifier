import numpy as np
from mnistClassifier import *
from utils import *

if __name__=="__main__":
    X, y = load_mnist()
    classifier = MnistClassifier([800], input_size=28*28, num_of_classes=10, weight_scale=5e-2,
                                 num_of_epochs=20, lr=0.0001, lr_decay=0.9)

    classifier.train(X, y)