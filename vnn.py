
"""
https://www.tensorflow.org/tutorials/quickstart/beginner

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

classifier.fit(data, labels, batch_size=100, steps=1000)

classifier.evaluate(test_data, test_labels)

# TODO: add Bias
"""
import numpy as np
from numpy.typing import NDArray
from typing import List

from abstractions import ModelData, ModelLabel
from constants import (
    OPTIMIZER_SGD, EDGE_DEFAULT_WEIGHT,
)
from utils import sigmoid, mean_squared_error


class Perceptron:
    def __init__(self, activation_function=sigmoid,
                 inbound_edges=None, outbound_edges=None):
        self.activation_function = activation_function
        self.inbound_edges = inbound_edges or []
        self.outbound_edges = outbound_edges or []
        # TODO: check if the correct terminology
        self.activation_value = None

    def refresh_value(self):
        """
        update value based on weighted sum of inbound edges & perceptrons
        :return:
        """
        weighted_avg = sum([e.weight * e.src_perceptron.activation_value
                           for e in self.inbound_edges]) / len(self.inbound_edges)

        self.activation_value = self.activation_function(weighted_avg)


class Edge:
    def __init__(self, src_perceptron=None, dst_perceptron=None, weight=EDGE_DEFAULT_WEIGHT):
        self.src_perceptron = src_perceptron
        self.dst_perceptron = dst_perceptron
        self.weight = weight


class Layer:
    def __init__(self, num_of_perceptrons: int):
        self.perceptrons = [Perceptron()] * num_of_perceptrons
        # self.bias = None

    # def add_bias(self):
    #     self.bias = BiasNode()

    def connect_to_next(self, layer):
        """
        connect layer to next layer (create relevant edges)
        :param layer: Layer object that comes after
        """
        raise NotImplementedError

    def refresh_values(self):
        """
        update layer's perceptrons value
        """
        [p.refresh_value() for p in self.perceptrons]

    def set_layer_data(self, data: NDArray[float]):
        """
        change whole layer's data (useful for input layers)
        :param data: array of data values
        """
        if len(data) != len(self.perceptrons):
            raise ValueError(f"data length ({len(data)}) and layer's "
                             f"perceptrons number ({len(self.perceptrons)}) "
                             f"do not match")
        for i in range(len(self.perceptrons)):
            self.perceptrons[i].activation_value = data[i]

    def values(self):
        """
        :return: layer's perceptrons values
        """
        return np.array([p.activation_value for p in self.perceptrons])


class DenseLayer(Layer):
    def connect_to_next(self, layer):
        """
        connect layer to next layer (create all possible edges - DENSE)
        :param layer: Layer object
        """
        # self_perceptrons = self.perceptrons + [self.bias] \
        #     if self.bias else self.perceptrons
        for src_perceptron in self.perceptrons:
            for dst_perceptron in layer.perceptrons:
                e = Edge(src_perceptron, dst_perceptron)
                src_perceptron.outbound_edges.append(e)
                dst_perceptron.inbound_edges.append(e)


class SequentialModel:
    def __init__(self, layers: List[Layer]):  # TODO: check how to add abstraction of inheriting classes
        self.optimizer = None
        self.loss_function = None
        self.layers = layers
        # self.layers[0].add_bias()
        self._connect_layers()

    def compile(self, optimizer=OPTIMIZER_SGD,
                loss_function=mean_squared_error):
        self.optimizer = optimizer
        self.loss_function = loss_function

    def _connect_layers(self):
        """
        add edges between adjacent layers
        """
        for i in range(len(self.layers) - 1):
            self.layers[i].connect_to_next(self.layers[i + 1])

    @staticmethod
    def _split_to_batches(data: ModelData, labels: ModelLabel, batch_size: int):
        """
        generator that splits the data & labels to batches, based on batch_size
        :param data: model's input data
        :param labels: matching labels
        :param batch_size: chunk size to split by
        :return: each time the next (data, labels)
        """
        i = 0
        while i < len(data):
            yield data[i: i + batch_size], labels[i: i + batch_size]
            i += batch_size

    def fit(self, data: ModelData, labels: ModelLabel,
            epochs: int, batch_size: int = 1):
        """
        :param data: input layer's data
        :param labels: expected output/result
        :param batch_size: chunk size to split the data to
        :param epochs: number of training iterations
        """
        if len(data) != len(labels):
            raise ValueError(f"data length ({len(data)}) and labels length"
                             f" ({len(labels)}) do not match")

        for _ in range(epochs):
            for batch_data, batch_labels in self._split_to_batches(data, labels, batch_size):
                pass
                # self.layers[0].set_layer_data(data[0]) # TODO: consider removal/fix
                # for each data input, create the relevant function full loss function
                # avg all of them into a new function
                # get gradient descent vector
                # adjust the weights accordingly, using the LEARNING_RATE

    def predict(self, data: ModelData) -> NDArray:
        """
        predict result based on provided data
        :param data:
        :return: array of predictions
        """
        predictions = []

        first_layer, other_layers = self.layers[0], self.layers[1:]
        for model_input in data:
            first_layer.set_layer_data(model_input)
            for layer in other_layers:
                layer.refresh_values()

            predictions.append(self.layers[-1].values())

        return np.array(predictions)


if __name__ == "__main__":
    # simulate & predict the AND bitwise operator function
    layers = [
        DenseLayer(2),
        DenseLayer(1),
    ]
    training_data = [
        [0, 0], [0, 1], [1, 0], [1, 1]
    ]
    training_labels = [
        [0], [0], [0], [1]
    ]

    test_data = [
        [0, 1], [1, 1]
    ]

    model = SequentialModel(layers)
    model.compile()
    model.fit(np.array(training_data), np.array(training_labels), epochs=5, batch_size=2)
    res = model.predict(np.array(test_data))
    print(res)
