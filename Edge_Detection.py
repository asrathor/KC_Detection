import numpy as np
import cv2

#To load the image, provide the file name or its path below.
img2_gray = cv2.imread('task1.png',0)

#Kernel for x axis
Gx = [[0 for x in range(3)] for y in range(3)]
Gx[0][0] = -1
Gx[0][1] = 0
Gx[0][2] = 1
Gx[1][0] = -2
Gx[1][1] = 0
Gx[1][2] = 2
Gx[2][0] = -1
Gx[2][1] = 0
Gx[2][2] = 1

#Kernel for y axis
Gy = [[0 for x in range(3)] for y in range(3)]
Gy[0][0] = -1
Gy[0][1] = -2
Gy[0][2] = -1
Gy[1][0] = 0
Gy[1][1] = 0
Gy[1][2] = 0
Gy[2][0] = 1
Gy[2][1] = 2
Gy[2][2] = 1

#Pad the matrix by 0
def padding(matrixx):

    final_mat = np.zeros((matrixx.shape[0]+2,matrixx.shape[1]+2))

    for i in range(matrixx.shape[0]):
        for j in range(matrixx.shape[1]):
            final_mat[i+1][j+1] = matrixx[i][j]

    return final_mat

#The image is convoluted (flipped)
def convolution(matrixx):

    conv_mat = np.zeros((3,3))
    conv_mat[0][0] = matrixx[2][2]
    conv_mat[0][1] = matrixx[2][1]
    conv_mat[0][2] = matrixx[2][0]
    conv_mat[1][0] = matrixx[1][2]
    conv_mat[1][2] = matrixx[1][0]
    conv_mat[2][0] = matrixx[0][2]
    conv_mat[2][1] = matrixx[0][1]
    conv_mat[2][2] = matrixx[0][0]

    return conv_mat

#From a large matrix, a (3,3) matrix is sliced
def slice_mat(matrixx,index_x,index_y):

    mat = np.zeros((3,3))
    mat[0][0] = matrixx[index_x][index_y]
    mat[0][1] = matrixx[index_x][index_y+1]
    mat[0][2] = matrixx[index_x][index_y+2]
    mat[1][0] = matrixx[index_x+1][index_y]
    mat[1][1] = matrixx[index_x+1][index_y+1]
    mat[1][2] = matrixx[index_x+1][index_y+2]
    mat[2][0] = matrixx[index_x+2][index_y]
    mat[2][1] = matrixx[index_x+2][index_y+1]
    mat[2][2] = matrixx[index_x+2][index_y+2]

    return mat

#Implemenation of the main Sobel Operator
def sobel_operator():

    rows = img2_gray.shape[0]+2
    columns = img2_gray.shape[1]+2
    comb_mat = np.zeros((rows,columns))
    x_img = np.zeros((rows,columns))
    y_img = np.zeros((rows,columns))
    img2_gray_padded = padding(img2_gray)

    for i in range(rows-2):
        for j in range(columns-2):

            slice_matrixx = slice_mat(img2_gray_padded,i,j)
            S1 = sum(sum(convolution(Gx) * slice_matrixx))
            S2 = sum(sum(convolution(Gy) * slice_matrixx))

            x_img[i+1][j+1] = S1
            y_img[i+1][j+1] = S2
            comb_mat[i + 1][j + 1] = ((S1 ** 2) + (S2 ** 2)) ** 0.5

    final_x_img = x_img/255
    final_y_img = y_img/255
    final_comb_img = comb_mat/255

    cv2.namedWindow('Edge_X', cv2.WINDOW_NORMAL)
    cv2.imshow('Edge_X', final_x_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow('Edge_Y', cv2.WINDOW_NORMAL)
    cv2.imshow('Edge_Y', final_y_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow('Magnitude_Edges', cv2.WINDOW_NORMAL)
    cv2.imshow('Magnitude_Edges', final_comb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sobel_operator()