import sys
import numpy as np
from sklearn.model_selection import KFold
from scipy.special import expit
from scipy.special import softmax


def one_hot_label(num):
    # convert y to vector which represent it
    v = np.zeros([10, 1])
    v[num] = 1
    return v


def get_TP(X_test, y_test, W1, W2, b1, b2):
    # count the amount of true positive
    TP = 0
    for x, y in zip(X_test, y_test):
        x.shape = (784, 1)
        x = x / 255
        z1 = np.dot(W1, x) + b1
        h1 = expit(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)
        max = 0
        save = 0
        for i in range(10):
            if ((h2[i]) > max):
                save = i
                max = np.sum(h2[i])
        if save == y:
            TP += 1
    return TP


def test(X_test, W1, W2, b1, b2):
    # write thr predictions to output file
    output_file = open("test_y", "w")
    for x in X_test:
        x.shape = (784, 1)
        x = x / 255
        z1 = np.dot(W1, x) + b1
        h1 = expit(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)
        max = 0
        save = 0
        for i in range(10):
            if ((h2[i]) > max):
                save = i
                max = np.sum(h2[i])
        output_file.write(str(save) + "\n")


def train(data_train_x, data_train_y):
    # load data from train_x.txt, train_y.txt and test_x.txt files
    learning_rate = 0.05
    neurons = 34
    W1 = np.random.rand(neurons, 784) / np.sqrt(784)
    b1 = np.random.rand(neurons, 1) / np.sqrt(784)
    W2 = np.random.rand(10, neurons) / np.sqrt(neurons)
    b2 = np.random.rand(10, 1) / np.sqrt(neurons)
    kf = KFold(n_splits=7)
    for train_index, test_index in kf.split(data_train_x):
        X_train, X_test = data_train_x[train_index], data_train_x[test_index]
        y_train, y_test = data_train_y[train_index], data_train_y[test_index]
        for i in range(50):
            # shuffle X_train and y_train
            indices_train = np.arange(X_train.shape[0])
            np.random.shuffle(indices_train)
            X_train = X_train[indices_train]
            y_train = y_train[indices_train]
            for x, y in zip(X_train, y_train):
                # normalize data
                x.shape = (784, 1)
                x = x / 255
                # forward
                z1 = np.dot(W1, x) + b1
                h1 = expit(z1)
                z2 = np.dot(W2, h1) + b2
                h2 = softmax(z2)
                # backward
                dz2 = (h2 - one_hot_label(y))  # dL/dz2
                dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
                db2 = dz2
                dH1 = np.dot(W2.T, dz2)
                dZ1 = dH1 * expit(z1) * (1 - expit(z1))
                dW1 = np.dot(dZ1, x.T)
                db1 = dZ1
                W2 = W2 - learning_rate * dW2
                b2 = b2 - learning_rate * db2
                W1 = W1 - learning_rate * dW1
                b1 = b1 - learning_rate * db1
    return W1, W2, b1, b2


if __name__ == '__main__':
    data_train_x = np.genfromtxt(sys.argv[1], delimiter=" ", dtype=np.float)
    data_train_y = np.genfromtxt(sys.argv[2], delimiter=' ', dtype=np.int)
    data_test_x = np.genfromtxt(sys.argv[3], delimiter=' ', dtype=np.float)
    W1, W2, b1, b2 = train(data_train_x, data_train_y)
    test(data_test_x, W1, W2, b1, b2)
    """
     # Initialize random parameters and inputs
    # load data from train_x.txt, train_y.txt and test_x.txt files
    learning_rate = 0.05
    neurons = 34
    W1 = np.random.rand(neurons, 784) / np.sqrt(784)
    b1 = np.random.rand(neurons, 1) / np.sqrt(784)
    W2 = np.random.rand(10, neurons) / np.sqrt(neurons)
    b2 = np.random.rand(10, 1)/ np.sqrt(neurons)
    kf = KFold(n_splits=7)
    TP = 0
    sum_TP = 0
    for train_index, test_index in kf.split(data_train_x):
        X_train, X_test = data_train_x[train_index], data_train_x[test_index]
        y_train, y_test = data_train_y[train_index], data_train_y[test_index]
        # shuffle X_test and y_test
        indices_test = np.arange(X_test.shape[0])
        np.random.shuffle(indices_test)
        X_test = X_test[indices_test]
        y_test = y_test[indices_test]
        for i in range(50):
            # shuffle X_train and y_train
            indices_train = np.arange(X_train.shape[0])
            np.random.shuffle(indices_train)
            X_train = X_train[indices_train]
            y_train = y_train[indices_train]
            for x, y in zip(X_train, y_train):
                m = X_train.shape[0]
                # normalize data
                x.shape = (784, 1)
                x = x /255
                # forward
                z1 = np.dot(W1, x) + b1
                h1 = expit(z1)
                z2 = np.dot(W2, h1) + b2
                h2 = softmax(z2)
                # backward
                dz2 = (h2 - one_hot_label(y))  # dL/dz2
                dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
                db2 = dz2
                dH1 = np.dot(W2.T, dz2)
                dZ1 = dH1 * expit(z1) * (1 - expit(z1))
                dW1 = np.dot(dZ1, x.T)
                db1 = dZ1
                W2 = W2 - learning_rate * dW2
                b2 = b2 - learning_rate * db2
                W1 = W1 - learning_rate * dW1
                b1 = b1 - learning_rate * db1
        TP = get_TP(X_test, y_test, W1, W2, b1, b2)
        TP = TP / y_test.shape[0]
        sum_TP += TP
    print(sum_TP/7)
"""



