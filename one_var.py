import numpy as np
# from numpy.linalg import inv
import matplotlib.pyplot as plt
 
 
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


mat = load_data("ex1data1.txt")
m, para = matrix_size(mat)
print_plot(mat)
