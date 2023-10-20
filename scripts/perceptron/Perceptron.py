import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self, num_inputs=2, learning_rate=0.01):
        self.num_inputs = num_inputs
        self.weights = np.ones(self.num_inputs)
        self.learning_rate = learning_rate
        self.bias = 1

    def weighted_sum(self, x):
        sum = 0
        for i in range(len(self.weights)):
            sum += self.weights[i] * x[i]
        return sum

    def activation_fn(self, x):
        if self.weighted_sum(x) + self.bias > 0:
            return 1
        return 0

    def infer(self, X):
        labels = []
        for x in X:
            labels.append(self.activation_fn(x))
        return labels

    def adjust_weights(self, x, y):
        self.bias = self.bias + self.learning_rate * (y - self.activation_fn(x))
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learning_rate * (y - self.activation_fn(x)) * x[i]

    def train(self, X, y, epochs=1000):
        for e in range(epochs):
            for i in range(len(X)):
                self.adjust_weights(X[i], y[i])


def generate():
    return make_blobs(n_samples=100, n_features=2, centers=2)


def synthetic():
    # generate synthetic data
    X, y = generate()
    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # train perceptron
    perceptron = Perceptron()
    perceptron.train(X_train, y_train)
    predict = perceptron.infer(X_test)

    # calculate wrong predictions
    wrong_pred_list = [predict != y_test]
    wrong_pred_total = np.sum(wrong_pred_list)
    print(f"Anteil der falsch klassifizierten syntetischen Daten: {(wrong_pred_total/len(y_test)) * 100} %")


def iris():
    # load iris data
    data = load_iris()
    X = data.data[:100, :]
    y = data.target[:100]

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # train perceptron
    perc = Perceptron()
    perc.train(X_train, y_train)
    y_pred = perc.infer(X_test)

    # calculate wrong predictions
    wrong_pred_list = [y_pred != y_test]
    wrong_pred= np.sum(wrong_pred_list)
    print(f"Anteil der falsch klassifizierten iris Daten: {(wrong_pred/len(y_test)) * 100} %")


if __name__ == '__main__':

    synthetic()
    iris()