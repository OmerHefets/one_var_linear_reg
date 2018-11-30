import numpy as np
# from numpy.linalg import inv
import matplotlib.pyplot as plt
# from sklearn import preprocessing


# one variable linear regression without using feature normalization


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


def print_reg(matrix, theta):
    x_arr = turn_mat_col_to_arr(matrix, 1)
    y_arr = turn_mat_col_to_arr(matrix, 2)
    plt.scatter(x_arr, y_arr)
    plt.xlabel('population')
    plt.ylabel('profit')
    x = np.arange(min(x_arr), max(x_arr), 0.2)
    plt.plot(x, theta[0] + theta[1] * x)
    plt.show()


def scaling_normalization(matrix):
    scaled_mat = matrix
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
    return rand_theta


def data_structuring(file):
    data = load_data(file)
    # scaling_normalization(data)
    m, parameters = matrix_size(data)
    x_data, y_data = seperate_x_y(data, parameters)
    x_data = add_x0_column(x_data)
    first_theta = random_theta(parameters)
    print_plot(data)
    return x_data, y_data, first_theta


def regression(file_name, alpha=0.01, iter=2000):
    x, y, theta = data_structuring(file_name)
    m = np.size(x, 0)
    error = []
    for i in range(iter):
        error.append(cost_function(x, y, theta))
        grad_calculation = np.dot(x.transpose(), h_sub_y(x, y, theta))
        theta -= (alpha / m) * grad_calculation
    plt.plot(error)
    plt.ylabel('J')
    plt.show()
    return theta


reg_theta = regression('ex1data1.txt')
mat = load_data('ex1data1.txt')
print_reg(mat, reg_theta)
