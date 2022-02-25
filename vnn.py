import copy
import numdifftools as nd  # stackoverflow.com/questions/65745683/how-to-install-scipy-on-apple-silicon-arm-m1
import numpy as np
from numpy.typing import NDArray
from typing import List, Callable

from abstractions import ModelData, ModelDataSingle, ModelLabels
from constants import EDGE_DEFAULT_WEIGHT, LEARNING_RATE
from utils import sigmoid, mean_squared_error


class Perceptron:
    def __init__(self, activation_function=sigmoid,
                 inbound_edges=None, outbound_edges=None):
        self.activation_function = activation_function
        self.inbound_edges = inbound_edges or []
        self.outbound_edges = outbound_edges or []
        self.activation_value = None

    def refresh_value(self):
        """
        update value based on weighted sum of inbound edges & perceptrons
        """
        weighted_sum = sum([e.weight * e.src_perceptron.activation_value
                           for e in self.inbound_edges])

        self.activation_value = self.activation_function(weighted_sum)


class Edge:
    def __init__(self, src_perceptron=None, dst_perceptron=None, weight=EDGE_DEFAULT_WEIGHT):
        self.src_perceptron = src_perceptron
        self.dst_perceptron = dst_perceptron
        self.weight = weight


class Layer:
    def __init__(self, num_of_perceptrons: int):
        self.perceptrons = [Perceptron() for _ in range(num_of_perceptrons)]

    def __len__(self):
        return len(self.perceptrons)

    def connect_to_next(self, layer):
        """
        connect layer to next layer (create relevant edges)
        :param layer: Layer object that comes after
        """
        raise NotImplementedError

    @property
    def weights(self):
        return [e.weight for p in self.perceptrons for e in p.outbound_edges]

    def set_weights(self, weights):
        i = 0
        for p in self.perceptrons:
            for edge in p.outbound_edges:
                edge.weight = weights[i]
                i += 1

    def refresh_values(self):
        """
        update layer's perceptrons value
        """
        [p.refresh_value() for p in self.perceptrons]

    def set_layer_data(self, first_layer_data: ModelDataSingle):
        """
        change whole layer's data (useful for input layers)
        :param first_layer_data: array of data values
        """
        if len(first_layer_data) != len(self.perceptrons):
            raise ValueError(f"data length ({len(first_layer_data)}) and layer's "
                             f"perceptrons number ({len(self.perceptrons)}) "
                             f"do not match")
        for i in range(len(self.perceptrons)):
            self.perceptrons[i].activation_value = first_layer_data[i]

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
        for src_perceptron in self.perceptrons:
            for dst_perceptron in layer.perceptrons:
                e = Edge(src_perceptron, dst_perceptron)
                src_perceptron.outbound_edges.append(e)
                dst_perceptron.inbound_edges.append(e)


class DenseInputLayer(Layer):
    def __init__(self, num_of_perceptrons: int):
        super().__init__(num_of_perceptrons)
        self.bias = Perceptron()
        self.bias.activation_value = 1

    def __len__(self):
        return super().__len__() + 1

    @property
    def weights(self):
        return [e.weight for p in [self.bias] + self.perceptrons for e in p.outbound_edges]

    def set_weights(self, weights):
        i = 0
        for p in [self.bias] + self.perceptrons:
            for edge in p.outbound_edges:
                edge.weight = weights[i]
                i += 1

    def connect_to_next(self, layer):
        """
        connect layer to next layer (create all possible edges - DENSE)
        :param layer: Layer object
        """
        for src_perceptron in [self.bias] + self.perceptrons:
            for dst_perceptron in layer.perceptrons:
                e = Edge(src_perceptron, dst_perceptron)
                src_perceptron.outbound_edges.append(e)
                dst_perceptron.inbound_edges.append(e)


class SequentialModelSGD:
    def __init__(self, layers: List[Layer]):
        self.loss_function = None
        self.layers = layers
        self._connect_layers()

    def compile(self,
                loss_function=mean_squared_error):
        self.loss_function = loss_function

    def _connect_layers(self):
        """
        add edges between adjacent layers
        """
        for i in range(len(self.layers) - 1):
            self.layers[i].connect_to_next(self.layers[i + 1])

    @staticmethod
    def _split_to_batches(data: ModelData, labels: ModelLabels, batch_size: int):
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

    def _set_weights(self, weights):
        i = 0
        for layer in self.layers:
            layer_weights_len = len(layer.weights)
            layer.set_weights(weights[i: i + layer_weights_len])
            i += layer_weights_len

    @property
    def _weights(self):
        weights = []
        for layer in self.layers:
            weights.extend(layer.weights)
        return weights

    def _create_batch_function(self, data: ModelData, labels: ModelLabels) -> Callable:
        """
        :param data: x input
        :param labels: y output
        :return: the total loss function for the batch
        """
        model_copy = copy.deepcopy(self)

        def batch_function(weights):
            model_copy._set_weights(weights)

            loss_sum = 0
            total_loss_items = 0
            for i, input_layer_data in enumerate(data):
                model_copy._propagate_forward(input_layer_data)

                for j, activation_val in enumerate(model_copy.layers[-1].values()):
                    loss_sum += model_copy.loss_function(activation_val, labels[i][j])
                    total_loss_items += 1

            return loss_sum / total_loss_items

        return batch_function

    def fit(self, data: ModelData, labels: ModelLabels,
            epochs: int, batch_size: int = 1):
        """
        train model based on SGD
        https://en.wikipedia.org/wiki/Stochastic_gradient_descent

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
                batch_func: Callable = self._create_batch_function(batch_data, batch_labels)
                weights = self._weights
                gradient = nd.Gradient(batch_func)(weights)

                updated_weights = []
                for i in range(len(weights)):
                    updated_weights.append(weights[i] - LEARNING_RATE * gradient[i])

                self._set_weights(updated_weights)

    def _propagate_forward(self, input_layer_data: ModelDataSingle) -> None:
        """
        propagate model forward based on input for the first layer
        :param input_layer_data: first layer input
        """
        first_layer, other_layers = self.layers[0], self.layers[1:]
        first_layer.set_layer_data(input_layer_data)
        for layer in other_layers:
            layer.refresh_values()

    def predict(self, data: ModelData) -> NDArray:
        """
        predict result based on provided data
        :param data:
        :return: array of predictions
        """
        predictions = []

        for input_layer_data in data:
            self._propagate_forward(input_layer_data)
            predictions.append(self.layers[-1].values())

        return np.array(predictions)


if __name__ == "__main__":
    # simulate & predict the AND bitwise operator function
    layers = [
        DenseInputLayer(2),
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

    model = SequentialModelSGD(layers)
    model.compile()
    model.fit(np.array(training_data), np.array(training_labels), batch_size=4, epochs=50000)
    res = model.predict(np.array(test_data))
    print(res)  # expected: [0, 1]
