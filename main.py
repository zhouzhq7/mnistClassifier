import numpy as np
from mnistClassifier import *
from utils import *
import sys
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

hidden_dims = [500, 300]
default_model_path = './model/checkpoint.pkl'


def train():
    X_dict, y_dict = load_mnist()
    classifier = MnistClassifier(hidden_dims=hidden_dims, input_size=28*28, num_of_classes=10, weight_scale=5e-2,
                                 num_of_epochs=50, lr=0.001, lr_decay=0.99,
                                 verbose_every=0, batch_size=16)
    classifier.train(X_dict=X_dict, y_dict=y_dict)

    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    plt.subplot(121)
    plt.plot(classifier.loss_history, linewidth=3.0)
    plt.ylim((0, 10))
    plt.xlabel('iterations')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.plot(classifier.train_accuracy_his, label='train', linewidth=3.0)
    plt.plot(classifier.val_accurracy_his, label='test', linewidth=3.0)
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('./figures/train.png')

def test():
    X_dict, y_dict = load_mnist()
    classifier = MnistClassifier(hidden_dims=hidden_dims, printable=False)
    classifier.load_checkpoint(default_model_path)
    test_acc = classifier.cal_accuracy(X_dict['val'], y_dict['val'])
    print ("Test accuracy is : %8f" % test_acc)

def inference(img_path, printable=False, save_to_file=True):
    img_names, images = load_images_from_directory(img_path)
    classifier = MnistClassifier(hidden_dims=hidden_dims, printable=False)
    classifier.load_checkpoint(default_model_path)
    y_pred = classifier.inference(images)

    if printable:
        for i in range(len(img_names)):
            print ("{} : {}".format(img_names[i], y_pred[i]))

    if save_to_file:
        pred_file = './prediction.txt'

        with open(pred_file, 'w') as f:
            f.write("%-15s %-15s\n" %('Name', 'Predict label'))
            for i in range(len(img_names)):
                f.write('%-15s %-2d\n' % (img_names[i], y_pred[i]))


if __name__=="__main__":
    n = len(sys.argv)

    if n == 1:
        train()
    elif n > 1:
        mode = sys.argv[1]
        if mode == 'train':
            train()
        elif mode == 'test':
            test()
        elif mode == 'inference':
            if n <= 2:
                raise Exception('Please give the image dir.')
            else:
                path = sys.argv[2]
                printable, save_to_file = False, True
                if n == 4:
                    printable = sys.argv[3]
                if n == 5:
                    save_to_file = sys.argv[4]
                inference(path, printable, save_to_file)
        else:
            raise Exception('Mode {} is not supported.'.format(sys[1]))
