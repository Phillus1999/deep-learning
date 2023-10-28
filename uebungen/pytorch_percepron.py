import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class Perceptron:
    # Perzeptron implementierung mit pytorch
    def __init__(self, num_inputs=2, learning_rate=0.01):
        self.num_inputs = num_inputs
        self.weights = torch.ones(self.num_inputs)
        self.learning_rate = learning_rate
        self.bias = 1

    def calculate_labels(self, X):
        """
        berechnet die prediction des perceptron
        :param X: data tensor
        :return: label tensor
        """
        # torch.sign gibt -1 oder 1 zurück je nachdem ob das argument negativ oder positiv ist
        # und 0 falls das Argument genau 0 ist.
        # TODO: FRAGE AN KORREKTUR: nutzt also gradient descent?
        return torch.sign(torch.matmul(X, self.weights) + self.bias)

    def train(self, X, y, epochs=50):
        """
        Passt die Gewichte und das bias des perceptron an
        :param X: data tensor
        :param y: label tensor
        :param epochs: anzahl der durchläufe
        :return: None
        """
        for epoch in range(epochs):
            y_pred = self.calculate_labels(X)
            error = y - y_pred

            self.weights = self.weights + self.learning_rate * torch.matmul(error, X)
            self.bias = self.bias + self.learning_rate * torch.sum(error)


if __name__ == '__main__':

    data = make_moons(n_samples=100)

    # erstelle die tensoren aus den numpy arrays
    features = torch.from_numpy(data[0]).float()
    labels = torch.from_numpy(data[1]).float()

    # erstelle das perceptron
    perc = Perceptron()

    # trainiere das perceptron
    perc.train(features, labels)

    # erstelle einen Plot, um das Ergebnis zu visualisieren
    fig = plt.figure(figsize=(12, 5))
    # setze x und y limits
    xlim = (features[:, 0].min() - 0.1, features[:, 0].max() + 0.1)
    ylim = (features[:, 1].min() - 0.1, features[:, 1].max() + 0.1)

    # plotte die daten mit den richtigen labels
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.scatter(features[:, 0], features[:, 1], c=labels)
    plt.title('Original Labels')

    # plotte die daten mit den vorhergesagten labels
    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.scatter(features[:, 0], features[:, 1], c=perc.calculate_labels(features))
    plt.title('Predicted Labels')

    # plotte die entscheidungsgrenze des perceptrons
    x1 = torch.tensor([features[:, 0].min() -0.1 , features[:, 0].max()])
    x2 = -(perc.weights[0] * x1 + perc.bias) / perc.weights[1]
    plt.plot(x1, x2, '--r')

    plt.show()
