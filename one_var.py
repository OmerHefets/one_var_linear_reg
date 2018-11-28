import numpy as np
# from numpy.linalg import inv
import matplotlib.pyplot as plt
# from sklearn import preprocessing


def load_data(file_name):
    matrix = np.loadtxt(file_name, delimiter=",")
    return matrix


def seperate_x_y(matrix, parameters):
    x = matrix[:, 0:(parameters-1)]
    y = matrix[:, (parameters-1):parameters]
    return x, y


def matrix_size(matrix):
    rows = np.size(matrix, 0)
    columns = np.size(matrix, 1)
    return rows, columns


def turn_mat_col_to_arr(matrix, col_number):
    rows, columns = matrix_size(matrix)
    arr = []
    if col_number > columns:
        print("Error! the column does not exist")
        exit(1)
    to_arr = matrix[:, (col_number-1):col_number]
    to_arr = np.squeeze(np.asarray(to_arr))
    for i in range(rows):
        arr.append(to_arr[i])
    return arr


def print_plot(matrix):
    x_arr = turn_mat_col_to_arr(matrix, 1)
    y_arr = turn_mat_col_to_arr(matrix, 2)
    plt.scatter(x_arr, y_arr)
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.show()


def scaling_normalization(matrix):
    scaled_mat = np.matrix(matrix)
    rows, columns = matrix_size(scaled_mat)
    for i in range(1, columns + 1):
        columns_arr = turn_mat_col_to_arr(scaled_mat, i)
        col_avg = np.average(columns_arr)
        col_std = np.std(columns_arr)
        scaled_mat[:, (i-1):i] -= col_avg
        scaled_mat[:, (i-1):i] /= col_std
    return scaled_mat


def h_sub_y(x, y, theta):
    return np.dot(x, theta) - y


def cost_function(x, y, theta):
    # m = #_of_examples
    m = np.size(x, 0)
    calculated_h = h_sub_y(x, y, theta)
    J = (1 / 2 * m) * np.dot(calculated_h.transpose(), calculated_h)
    return np.asscalar(J)


def add_x0_column(x_matrix):
    m = np.size(x_matrix, 0)
    x_matrix = np.c_[np.ones(m), x_matrix]
    return x_matrix


def random_theta(para_num):
    rand_theta = np.reshape(np.random.rand(para_num), (para_num, 1))
    print(rand_theta)


"""def regression(file, ):
    data = load_data(file)
    m, parameters = matrix_size(data)
    x_data, y_data = seperate_x_y(data, 1)
    x_data = add_x0_column(x_data)"""


mat = load_data("ex1data1.txt")
m, parameters = matrix_size(mat)
mat = add_x0_column(mat)
random_theta(4)
# x_para, y_para = seperate_x_y(mat, parameters)
# x_para = add_x0_column(x_para)
# print(cost_function(x_para, y_para, np.matrix('1;1')))

