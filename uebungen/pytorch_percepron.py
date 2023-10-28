import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class Perceptron:
    # implement perceptron using pytorch
    def __init__(self, num_inputs=2, learning_rate=0.01):
        self.num_inputs = num_inputs
        self.weights = torch.ones(self.num_inputs)
        self.learning_rate = learning_rate
        self.bias = 1

    def forward(self, X):
        """
        berechnet die prediction des perceptrons
        :param X: data tensor
        :return: label tensor
        """
        # TODO: verstehen warum das genau so funktioniert
        # hier ist mm nicht das gleiche wie matmul ?!
        # torch.sign gibt -1 oder 1 zurück je nachdem ob das argument negativ oder positiv ist
        # nutzt also gradient descent ?!
        # TODO: gradient descent verstehen
        return torch.sign(torch.matmul(X, self.weights) + self.bias)

    def train(self, X, y, epochs=10):
        """
        passt die gewichte und das bias des perceptrons an
        :param X: data tensor
        :param y: label tensor
        :param epochs: number of epochs
        :return: None
        """
        for epoch in range(epochs):
            # erstellt einen tensor der selben länge wie X und füllt ihn mit den predictions
            y_pred = self.forward(X)
            # erstellt einen tensor der selben länge wie y_pred und füllt ihn mit den fehlern
            error = y - y_pred
            # TODO: vielleicht hier auch vektorisieren und nicht iterativ ?
            for i in range(len(X)):
                # passe die gewichte an
                self.weights = self.weights + self.learning_rate * error[i] * X[i]
                # passe den bias an
                self.bias = self.bias + self.learning_rate * error[i]


if __name__ == '__main__':
    # TODO: darstellung der entscheidungsgrenze über epochen darstellen

    # erstelle die daten (2 features, 100 samples)
    data = make_moons(n_samples=100)

    # erstelle die tensoren aus den numpy arrays
    features = torch.from_numpy(data[0]).float()
    labels = torch.from_numpy(data[1]).float()

    # erstelle das perceptron
    perc = Perceptron()

    # trainiere das perceptron
    perc.train(features, labels, epochs=50)

    # erstelle einen Plot um das Ergebnis zu visualisieren
    fig = plt.figure(figsize=(12, 5))
    # setze x und y limits
    xlim = (features[:, 0].min() - 0.1, features[:, 0].max() + 0.1)
    ylim = (features[:, 1].min() - 0.1, features[:, 1].max() + 0.1)

    # plotte die daten mit richtigen labels
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.scatter(features[:, 0], features[:, 1], c=labels)
    plt.title('Original Labels')

    # plotte die daten mit den predictions
    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.scatter(features[:, 0], features[:, 1], c=perc.forward(features))
    plt.title('Predicted Labels')

    # plotte die entscheidungsgrenze
    x1 = torch.tensor([features[:, 0].min(), features[:, 0].max()])
    x2 = -(perc.weights[0] * x1 + perc.bias) / perc.weights[1]
    plt.plot(x1, x2, '--r')

    plt.show()
