import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class Perceptron:
    # implement perceptron using pytorch
    def __init__(self, num_inputs=2, learning_rate=0.01):
        self.num_inputs = num_inputs
        self.weights = torch.ones(self.num_inputs, 1)
        self.learning_rate = learning_rate
        self.bias = 1
    def infer(self, x):
        """
        :param x: input tensor
        :return: output tensor after activation_function
        """
        weighted_sum = torch.mm(x, self.weights) + self.bias
        return torch.sign(weighted_sum)

    def update_weights(self, x, y):
        """
        :param x: input tensor
        :param y: ground truth
        :param y_hat: prediction
        :return: None
        """
        y_pred = self.infer(x).squeeze(1)
        print(y_pred)
        print(y)
        error = y - y_pred
        # Update weights
        weight_update = torch.mm(x.T, error)  # shape (num_features, 1)
        self.weights += self.learning_rate * weight_update
        # Update bias
        bias_update = torch.sum(error, dim=0)  # shape (1,)
        self.bias += self.learning_rate * bias_update

    def train(self, x, y, epochs=10):
        """
        :param x: input tensor
        :param y: ground truth
        :param epochs: number of training epochs
        :return: None
        """
        for epoch in range(epochs):
            self.update_weights(x, y)

if __name__ == '__main__':
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Ground Truth")
    perceptron = Perceptron()
    perceptron.train(X, y)
    y_pred = perceptron.infer(X)
    plt.subplot(1,2,2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Prediction")
    plt.show()