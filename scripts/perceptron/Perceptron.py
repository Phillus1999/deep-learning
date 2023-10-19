import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    def adjust_weights(self, x, label):
        self.bias = self.bias + self.learning_rate * (label - self.activation_fn(x))
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learning_rate * (label - self.activation_fn(x)) * x[i]

    def train(self, X, y, epochs=2000):
        for e in range(epochs):
            labels = self.infer(X)
            for i in range(len(X)):
                self.adjust_weights(X[i], labels[i])


def generate():
    return make_blobs(n_samples=100, n_features=2, centers=2)


if __name__ == '__main__':
    def run_perceptron(X, y):
        # just for plotting
        x_min, x_max = min(X.T[0] - 1), max(X.T[0] + 1)
        y_min, y_max = min(X.T[1] - 1), max(X.T[1] + 1)

        fig = plt.figure(figsize=(10, 5))

        # plot correct class
        ax = plt.subplot(1, 2, 1)
        plt.title('Correct Class')
        plt.scatter(X[:, 0], X[:, 1], c=y)
        ax.set_xlim([x_min - 1, x_max + 1])
        ax.set_ylim([y_min - 1, y_max + 1])
        plt.axhline(y=0, color='gray', alpha=0.5, linestyle='dotted')
        plt.axvline(x=0, color='gray', alpha=0.5, linestyle='dotted')

        # create and train perceptron
        perc = Perceptron()
        perc.train(X, y)
        y_pred = perc.infer(X)

        # plot prediction
        ax = plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        ax.set_xlim([x_min - 1, x_max + 1])
        ax.set_ylim([y_min - 1, y_max + 1])
        plt.axhline(y=0, color='gray', alpha=0.5, linestyle='dotted')
        plt.axvline(x=0, color='gray', alpha=0.5, linestyle='dotted')

        # show decision boundary
        x1 = np.linspace(x_min, x_max)
        x2 = -(perc.weights[0] * x1 + perc.bias) / perc.weights[1]
        plt.plot(x1, x2, '--r')
        plt.show()
        return fig

    with PdfPages('perceptron/results.pdf') as pdf:
        for i in range(10):
            X, y = generate()
            fig = run_perceptron(X, y)
            pdf.savefig(fig)  # saves the current figure into the pdf
            plt.close(fig)  # close the figure to free up memory

