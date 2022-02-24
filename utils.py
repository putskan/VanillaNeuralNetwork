import math


def sigmoid(x: float):
    """
    sigmoid function
    :param x:
    :return: sigmoid function output
    """
    # TODO: make sure it's correct
    return 1 / (1 + math.exp(-x))


def mean_squared_error(x1: float, x2: float):
    return (x1 - x2) ** 2
