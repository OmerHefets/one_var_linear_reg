import numpy as np
# from numpy.linalg import inv
# import matplotlib.pyplot as plt


def load_data(file_name):
    matrix = np.loadtxt(file_name, delimiter=",")
    return matrix


def seperate_x_y(matrix, parameters):
    x = matrix[:, 0:(parameters-1)]
    y = matrix[:, (parameters-1):parameters]
    return x, y


def matrix_size(matrix):
    # number of training examples
    rows = np.size(matrix, 0)
    # number of parameters (including Y)
    columns = np.size(matrix, 1)
    return rows, columns


mat = load_data("ex1data1.txt")
m, para = matrix_size(mat)

