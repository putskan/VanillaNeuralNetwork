import math


def sigmoid(x: float):
    """
    calc Sigmoid
    https://en.wikipedia.org/wiki/Sigmoid_function
    sigmoid function
    :param x: func input
    :return: sigmoid function output
    """
    return 1 / (1 + math.exp(-x))


def mean_squared_error(x1: float, x2: float):
    """
    calc MSE
    https://en.wikipedia.org/wiki/Mean_squared_error
    :param x1: func input 1
    :param x2: func input 2
    :return:
    """
    return (x1 - x2) ** 2
