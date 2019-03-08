import numpy as np
import math
import cv2

#To load the image, provide the file name or its path below. (You have to provide the path many times below too)
img2_gray = cv2.imread('task2.jpg',0)

print(img2_gray.shape)
global counter
counter = 0
first_octave = []

#Below is the code for resizing
#Dividing the original image by 2
new_img1 = np.zeros((int(img2_gray.shape[0]/2),int(img2_gray.shape[1]/2)))
for i in range(new_img1.shape[0]):
    for j in range(new_img1.shape[1]):
        new_img1[i][j] = img2_gray[i*2+1][j*2+1]

#Dividing the previous image by 2
new_img2 = np.zeros((int(new_img1.shape[0]/2),int(new_img1.shape[1]/2)))
for i in range(new_img2.shape[0]):
    for j in range(new_img2.shape[1]):
        new_img2[i][j] = img2_gray[i*4+1][j*4+1]

#Dividing the previous image by again 2
new_img3 = np.zeros((int(new_img2.shape[0]/2),int(new_img2.shape[1]/2)))
for i in range(new_img3.shape[0]):
    for j in range(new_img3.shape[1]):
        new_img3[i][j] = img2_gray[i*8+1][j*8+1]

#Pad the matrix by 0. This function is not used since we need to pad by 3 layers on all sides.
def padding(matrixx):

    final_mat = np.zeros((matrixx.shape[0]+2,matrixx.shape[1]+2))

    for i in range(matrixx.shape[0]):
        for j in range(matrixx.shape[1]):
            final_mat[i+1][j+1] = matrixx[i][j]

    return final_mat

#Pad the matrix by 0, three layers on each side.
def padding_6(matrixx):

    final_mat = np.zeros((matrixx.shape[0] + 6, matrixx.shape[1] + 6))

    final_mat[3:-3,3:-3] = matrixx

    return final_mat

#To rotate the matrix by 90 degrees
def rotateMatrix(matrixx):
    size = matrixx.shape[0]
    for i in range(0, int(size/2)):
        for j in range(i, size-i-1):

            var = matrixx[i][j]
            matrixx[i][j] = matrixx[j][size-i-1]
            matrixx[j][size-i-1] = matrixx[size-i-1][size-j-1]
            matrixx[size-i-1][size-j-1] = matrixx[size-j-1][i]
            matrixx[size-j-1][i] = var

    return matrixx

#The implemenation of gaussian at a single matrix location
def gaussianBlur(sigma, x, y):

    sq_sig = sigma**2
    pi_sig = 2 * math.pi * sq_sig

    sq_x_y = (x**2) + (y**2)

    div_x_y = sq_x_y / (2 * (sigma**2))

    exp_x_y = np.exp(-div_x_y)

    gauss_result = exp_x_y / pi_sig

    return gauss_result

#Convolution is performed by rotating the matrix by 90 degrees two times
def convol(operator):

    convol_mat = rotateMatrix(rotateMatrix(operator))

    return convol_mat

#The gaussian kernel of size (7,7) is computed.
def kernel_gauss(sigma):
    kernel_size = 7
    operator_arr = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            if j == 0:
                operator_arr[i][j] = gaussianBlur(sigma,i-3,j+3)
            if j == 1:
                operator_arr[i][j] = gaussianBlur(sigma, i - 3, j + 1)
            if j == 2:
                operator_arr[i][j] = gaussianBlur(sigma, i - 3, j - 1)
            if j == 3:
                operator_arr[i][j] = gaussianBlur(sigma, i - 3, j - 3)
            if j == 4:
                operator_arr[i][j] = gaussianBlur(sigma, i - 3, j - 5)
            if j == 5:
                operator_arr[i][j] = gaussianBlur(sigma, i - 3, j - 7)
            if j == 6:
                operator_arr[i][j] = gaussianBlur(sigma, i - 3, j - 9)
    #We perform the convolution here itself.
    operator_arr_convl = convol(operator_arr)

    return operator_arr_convl

#The matrix is sliced into 7,7 from original image.
def slice_mat(matrixx,index_x,index_y):

    slice_size = 7
    mat = np.zeros((slice_size, slice_size))

    for i in range(slice_size):
        for j in range(slice_size):

            mat[i][j] = matrixx[index_x+i][index_y+j]

    return mat

#Implementation for generating an octave
def octave(sigma,count):

    rows = 0
    columns = 0
    img = []
    if count == 1:
        rows = img2_gray.shape[0]+6
        columns = img2_gray.shape[1]+6
        img = img2_gray

    if count == 2:
        rows = new_img1.shape[0]+6
        columns = new_img1.shape[1]+6
        img = new_img1

    if count == 3:
        rows = new_img2.shape[0]+6
        columns = new_img2.shape[1]+6
        img = new_img2

    if count == 4:
        rows = new_img3.shape[0]+6
        columns = new_img3.shape[1]+6
        img = new_img3

    out_img = np.zeros((rows,columns))
    print(out_img.shape)
    img2_gray_padded = padding_6(img)

    gauss = kernel_gauss(sigma)
    gauss_norm = gauss/sum(sum(gauss))
    print(sum(sum(gauss_norm)))

    for i in range(rows-6):
        for j in range(columns-6):
            mats = slice_mat(img2_gray_padded,i,j)
            out_img[i+3][j+3] = sum(sum(mats * gauss_norm))

    if count == 1:
        if counter == 1:
            cv2.imwrite('oct1_img1.png',out_img)
            print('A')
        if counter == 2:
            cv2.imwrite('oct1_img2.png', out_img)
            print('B')
        if counter == 3:
            cv2.imwrite('oct1_img3.png', out_img)
            print('C')
        if counter == 4:
            cv2.imwrite('oct1_img4.png', out_img)
            print('D')
        if counter == 5:
            cv2.imwrite('oct1_img5.png', out_img)
            print('E')
    if count == 2:
        if counter == 1:
            cv2.imwrite('oct2_img1.png',out_img)
            print('A')
        if counter == 2:
            cv2.imwrite('oct2_img2.png', out_img)
            print('B')
        if counter == 3:
            cv2.imwrite('oct2_img3.png', out_img)
            print('C')
        if counter == 4:
            cv2.imwrite('oct2_img4.png', out_img)
            print('D')
        if counter == 5:
            cv2.imwrite('oct2_img5.png', out_img)
            print('E')
    if count == 3:
        if counter == 1:
            cv2.imwrite('oct3_img1.png',out_img)
            print('A')
        if counter == 2:
            cv2.imwrite('oct3_img2.png', out_img)
            print('B')
        if counter == 3:
            cv2.imwrite('oct3_img3.png', out_img)
            print('C')
        if counter == 4:
            cv2.imwrite('oct3_img4.png', out_img)
            print('D')
        if counter == 5:
            cv2.imwrite('oct3_img5.png', out_img)
            print('E')
    if count == 4:
        if counter == 1:
            cv2.imwrite('oct4_img1.png',out_img)
            print('A')
        if counter == 2:
            cv2.imwrite('oct4_img2.png', out_img)
            print('B')
        if counter == 3:
            cv2.imwrite('oct4_img3.png', out_img)
            print('C')
        if counter == 4:
            cv2.imwrite('oct4_img4.png', out_img)
            print('D')
        if counter == 5:
            cv2.imwrite('oct4_img5.png', out_img)
            print('E')

#The original function of creating octave is run 5 times for each sigma
def multiple_octave_1():

    sigma = np.asarray([(1/(2 ** 0.5)), 1, (2 ** 0.5), 2, (2 * (2 ** 0.5))],dtype=float)
    for i in range(5):
        global counter
        counter = counter + 1
        octave(sigma[i],1)

#The original function of creating octave is run 5 times for each sigma
def multiple_octave_2():
    global counter
    counter = 0
    sigma = np.asarray([(2 ** 0.5), 2, (2 * (2 ** 0.5)), 4, (4 * (2 ** 0.5))], dtype=float)
    for i in range(5):
        counter = counter + 1
        octave(sigma[i],2)

#The original function of creating octave is run 5 times for each sigma
def multiple_octave_3():
    global counter
    counter = 0
    sigma = np.asarray([(2 * (2 ** 0.5)), 4, (4 * (2 ** 0.5)), 8, (8 * (2 ** 0.5))], dtype=float)
    for i in range(5):
        counter = counter + 1
        octave(sigma[i],3)

#The original function of creating octave is run 5 times for each sigma
def multiple_octave_4():
    global counter
    counter = 0
    sigma = np.asarray([(4 * (2 ** 0.5)), 8, (8 * (2 ** 0.5)), 16, (16 * (2 ** 0.5))], dtype=float)
    for i in range(5):
        counter = counter + 1
        octave(sigma[i],4)

#Create octaves for all the blurs required
def create_all_octave():

    multiple_octave_1()
    multiple_octave_2()
    multiple_octave_3()
    multiple_octave_4()

#To compute DOGs for first octave
def DOG_1():

    oct1_img1 = cv2.imread('oct1_img1.png', 0)
    oct1_img2 = cv2.imread('oct1_img2.png', 0)
    oct1_img3 = cv2.imread('oct1_img3.png', 0)
    oct1_img4 = cv2.imread('oct1_img4.png', 0)
    oct1_img5 = cv2.imread('oct1_img5.png', 0)
    octave = [oct1_img1,oct1_img2,oct1_img3,oct1_img4,oct1_img5]

    for i in range(4):
        #Note that when performing subtraction, if result is negative then we need to ensure that array type is int16 which can store negative numbers.
        oct1 = octave[i].astype(np.int16)
        oct2 = octave[i+1].astype(np.int16)
        #Subtract two consecutive octaves
        temp = np.subtract(oct1, oct2)
        #Scale the values between 0,255
        temp = np.clip(temp,0,255)
        dog = temp.astype(np.int16)
        #Normalize to sharpen the edges shown. If not normalize, then the pixel values will not be very apparent
        dog = (dog - dog.min())/(dog.max()-dog.min())
        dog = dog * 255
        if i == 0:
            cv2.imwrite('oct1_dog1.png',dog)
        if i == 1:
            cv2.imwrite('oct1_dog2.png',dog)
        if i == 2:
            cv2.imwrite('oct1_dog3.png',dog)
        if i == 3:
            cv2.imwrite('oct1_dog4.png',dog)

#To compute DOGs for second octave
def DOG_2():

    oct2_img1 = cv2.imread('oct2_img1.png', 0)
    oct2_img2 = cv2.imread('oct2_img2.png', 0)
    oct2_img3 = cv2.imread('oct2_img3.png', 0)
    oct2_img4 = cv2.imread('oct2_img4.png', 0)
    oct2_img5 = cv2.imread('oct2_img5.png', 0)
    octave2 = [oct2_img1, oct2_img2, oct2_img3, oct2_img4, oct2_img5]

    for i in range(4):
        # Note that when performing subtraction, if result is negative then we need to ensure that array type is int16 which can store negative numbers.
        oct1 = octave2[i].astype(np.int16)
        oct2 = octave2[i + 1].astype(np.int16)
        # Subtract two consecutive octaves
        temp = np.subtract(oct1, oct2)
        # Scale the values between 0,255
        temp = np.clip(temp, 0, 255)
        dog = temp.astype(np.int16)
        # Normalize to sharpen the edges shown. If not normalize, then the pixel values will not be very apparent
        dog = (dog - dog.min()) / (dog.max() - dog.min())
        dog = dog * 255
        if i == 0:
            cv2.imwrite('oct2_dog1.png', dog)
        if i == 1:
            cv2.imwrite('oct2_dog2.png', dog)
        if i == 2:
            cv2.imwrite('oct2_dog3.png', dog)
        if i == 3:
            cv2.imwrite('oct2_dog4.png', dog)

#To compute DOGs for third octave
def DOG_3():

    oct3_img1 = cv2.imread('oct3_img1.png', 0)
    oct3_img2 = cv2.imread('oct3_img2.png', 0)
    oct3_img3 = cv2.imread('oct3_img3.png', 0)
    oct3_img4 = cv2.imread('oct3_img4.png', 0)
    oct3_img5 = cv2.imread('oct3_img5.png', 0)
    octave3 = [oct3_img1, oct3_img2, oct3_img3, oct3_img4, oct3_img5]

    for i in range(4):
        # Note that when performing subtraction, if result is negative then we need to ensure that array type is int16 which can store negative numbers.
        oct1 = octave3[i].astype(np.int16)
        oct2 = octave3[i + 1].astype(np.int16)
        # Subtract two consecutive octaves
        temp = np.subtract(oct1, oct2)
        # Scale the values between 0,255
        temp = np.clip(temp, 0, 255)
        dog = temp.astype(np.int16)
        # Normalize to sharpen the edges shown. If not normalize, then the pixel values will not be very apparent
        dog = (dog - dog.min()) / (dog.max() - dog.min())
        dog = dog * 255
        if i == 0:
            cv2.imwrite('oct3_dog1.png', dog)
        if i == 1:
            cv2.imwrite('oct3_dog2.png', dog)
        if i == 2:
            cv2.imwrite('oct3_dog3.png', dog)
        if i == 3:
            cv2.imwrite('oct3_dog4.png', dog)

#To compute DOGs for fourth octave
def DOG_4():

    oct4_img1 = cv2.imread('oct4_img1.png', 0)
    oct4_img2 = cv2.imread('oct4_img2.png', 0)
    oct4_img3 = cv2.imread('oct4_img3.png', 0)
    oct4_img4 = cv2.imread('oct4_img4.png', 0)
    oct4_img5 = cv2.imread('oct4_img5.png', 0)
    octave4 = [oct4_img1, oct4_img2, oct4_img3, oct4_img4, oct4_img5]

    for i in range(4):
        # Note that when performing subtraction, if result is negative then we need to ensure that array type is int16 which can store negative numbers.
        oct1 = octave4[i].astype(np.int16)
        oct2 = octave4[i + 1].astype(np.int16)
        # Subtract two consecutive octaves
        temp = np.subtract(oct1, oct2)
        # Scale the values between 0,255
        temp = np.clip(temp, 0, 255)
        dog = temp.astype(np.int16)
        # Normalize to sharpen the edges shown. If not normalize, then the pixel values will not be very apparent
        dog = (dog - dog.min()) / (dog.max() - dog.min())
        dog = dog * 255
        if i == 0:
            cv2.imwrite('oct4_dog1.png', dog)
        if i == 1:
            cv2.imwrite('oct4_dog2.png', dog)
        if i == 2:
            cv2.imwrite('oct4_dog3.png', dog)
        if i == 3:
            cv2.imwrite('oct4_dog4.png', dog)

#This function will slice the original matrix into (3,3)
def slice_mat_three(matrixx,index_x,index_y):

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

#To compare the 3x3x3 cube and determine that the center point is greatest or least
def comp_matrix_1(mat1,mat2,mat3):
    temp2 = np.zeros(8)
    temp2[0] = mat2[0][0]
    temp2[1] = mat2[0][1]
    temp2[2] = mat2[0][2]
    temp2[3] = mat2[1][0]
    temp2[4] = mat2[1][2]
    temp2[5] = mat2[2][0]
    temp2[6] = mat2[2][1]
    temp2[7] = mat2[2][2]

    temp = np.zeros(3)
    temp[0] = mat1.min()
    temp[1] = temp2.min()
    temp[2] = mat3.min()
    temp1 = np.zeros(3)
    temp1[0] = mat1.max()
    temp1[1] = temp2.max()
    temp1[2] = mat3.max()
    var_min = temp.min()
    var_max = temp1.max()

    if mat2[1][1] < var_min or mat2[1][1] > var_max:
        return 1
    return 0

#Generate keypoints for the first octave
def keypoint_1():

    oct1_dog1 = cv2.imread('oct1_dog1.png', 0)
    oct1_dog2 = cv2.imread('oct1_dog2.png', 0)
    oct1_dog3 = cv2.imread('oct1_dog3.png', 0)
    oct1_dog4 = cv2.imread('oct1_dog4.png', 0)

    oct1_dog1 = oct1_dog1[2:-2,2:-2]
    oct1_dog2 = oct1_dog2[2:-2, 2:-2]
    oct1_dog3 = oct1_dog3[2:-2, 2:-2]
    oct1_dog4 = oct1_dog4[2:-2, 2:-2]

    rows = img2_gray.shape[0]
    columns = img2_gray.shape[1]
    key_mat_1 = np.zeros((rows,columns))
    key_mat_2 = np.zeros((rows,columns))

    for i in range(rows-2):
        for j in range(columns-2):

            slice_matrix_2 = slice_mat_three(oct1_dog2,i,j)
            slice_matrix_1 = slice_mat_three(oct1_dog1,i,j)
            slice_matrix_3 = slice_mat_three(oct1_dog3,i,j)
            slice_matrix_4 = slice_mat_three(oct1_dog4,i,j)
            key_mat_1[i+1][j+1] = comp_matrix_1(slice_matrix_1,slice_matrix_2,slice_matrix_3)
            key_mat_2[i+1][j+1] = comp_matrix_1(slice_matrix_2,slice_matrix_3,slice_matrix_4)

    key_mat_final = key_mat_1+key_mat_2

    img = cv2.imread('task2.jpg')
    img = img/255
    print(key_mat_final.shape)
    for i in range(key_mat_final.shape[0]):
        for j in range(key_mat_final.shape[1]):
            if key_mat_final[i][j] >= 1:
                #We subtract the image by -1 because the keypoint matrix is padded but original image is not.
                img[i-1][j-1][:] = 1
    cv2.namedWindow('Keypoint_Octave1', cv2.WINDOW_NORMAL)
    cv2.imshow('Keypoint_Octave1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key_mat_final

#Generate keypoints on the second octave
def keypoint_2():

    oct2_dog1 = cv2.imread('oct2_dog1.png', 0)
    oct2_dog2 = cv2.imread('oct2_dog2.png', 0)
    oct2_dog3 = cv2.imread('oct2_dog3.png', 0)
    oct2_dog4 = cv2.imread('oct2_dog4.png', 0)

    oct2_dog1 = oct2_dog1[2:-2, 2:-2]
    oct2_dog2 = oct2_dog2[2:-2, 2:-2]
    oct2_dog3 = oct2_dog3[2:-2, 2:-2]
    oct2_dog4 = oct2_dog4[2:-2, 2:-2]

    rows = oct2_dog1.shape[0]
    columns = oct2_dog1.shape[1]
    key_mat_1 = np.zeros((rows, columns))
    key_mat_2 = np.zeros((rows, columns))

    for i in range(rows-2):
        for j in range(columns-2):

            slice_matrix_2 = slice_mat_three(oct2_dog2,i,j)
            slice_matrix_1 = slice_mat_three(oct2_dog1,i,j)
            slice_matrix_3 = slice_mat_three(oct2_dog3,i,j)
            slice_matrix_4 = slice_mat_three(oct2_dog4,i,j)
            key_mat_1[i+1][j+1] = comp_matrix_1(slice_matrix_1,slice_matrix_2,slice_matrix_3)
            key_mat_2[i+1][j+1] = comp_matrix_1(slice_matrix_2,slice_matrix_3,slice_matrix_4)

    key_mat_final = key_mat_1 + key_mat_2

    img = cv2.imread('task2.jpg')
    print(key_mat_final.shape)
    img = img / 255
    for i in range(key_mat_final.shape[0]-2):
        for j in range(key_mat_final.shape[1]-2):
            if key_mat_final[i][j] >= 1:
                # We subtract the image by -3 because the keypoint matrix is padded but original image is not and multiply by 2 because the original octave by divided by 2.
                img[i*2-3][j*2-3][:] = 1

    cv2.namedWindow('Keypoints Octave2', cv2.WINDOW_NORMAL)
    cv2.imshow('Keypoints Octave2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key_mat_final

#Generate keypoints on the third octave
def keypoint_3():

    oct3_dog1 = cv2.imread('oct3_dog1.png', 0)
    oct3_dog2 = cv2.imread('oct3_dog2.png', 0)
    oct3_dog3 = cv2.imread('oct3_dog3.png', 0)
    oct3_dog4 = cv2.imread('oct3_dog4.png', 0)

    oct3_dog1 = oct3_dog1[2:-2, 2:-2]
    oct3_dog2 = oct3_dog2[2:-2, 2:-2]
    oct3_dog3 = oct3_dog3[2:-2, 2:-2]
    oct3_dog4 = oct3_dog4[2:-2, 2:-2]

    rows = oct3_dog1.shape[0]
    columns = oct3_dog1.shape[1]
    key_mat_1 = np.zeros((rows, columns))
    key_mat_2 = np.zeros((rows, columns))

    for i in range(rows-2):
        for j in range(columns-2):

            slice_matrix_2 = slice_mat_three(oct3_dog2,i,j)
            slice_matrix_1 = slice_mat_three(oct3_dog1,i,j)
            slice_matrix_3 = slice_mat_three(oct3_dog3,i,j)
            slice_matrix_4 = slice_mat_three(oct3_dog4,i,j)
            key_mat_1[i+1][j+1] = comp_matrix_1(slice_matrix_1,slice_matrix_2,slice_matrix_3)
            key_mat_2[i+1][j+1] = comp_matrix_1(slice_matrix_2,slice_matrix_3,slice_matrix_4)

    key_mat_final = key_mat_1 + key_mat_2

    img = cv2.imread('task2.jpg')

    img = img / 255
    print(key_mat_final.shape)
    for i in range(key_mat_final.shape[0]-2):
        for j in range(key_mat_final.shape[1]-2):
            if key_mat_final[i][j] >= 1:
                # We subtract the image by -5 because the keypoint matrix is padded but original image is not and multiply by 4 because the original octave by divided by 4.
                img[i*4-5][j*4-5][:] = 1

    cv2.namedWindow('Keypoints Octave3', cv2.WINDOW_NORMAL)
    cv2.imshow('Keypoints Octave3', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key_mat_final

#Generate key points on the fourth octave
def keypoint_4():

    oct4_dog1 = cv2.imread('oct4_dog1.png', 0)
    oct4_dog2 = cv2.imread('oct4_dog2.png', 0)
    oct4_dog3 = cv2.imread('oct4_dog3.png', 0)
    oct4_dog4 = cv2.imread('oct4_dog4.png', 0)

    oct4_dog1 = oct4_dog1[2:-2, 2:-2]
    oct4_dog2 = oct4_dog2[2:-2, 2:-2]
    oct4_dog3 = oct4_dog3[2:-2, 2:-2]
    oct4_dog4 = oct4_dog4[2:-2, 2:-2]

    rows = oct4_dog1.shape[0]
    columns = oct4_dog1.shape[1]
    key_mat_1 = np.zeros((rows, columns))
    key_mat_2 = np.zeros((rows, columns))

    for i in range(rows-2):
        for j in range(columns-2):

            slice_matrix_2 = slice_mat_three(oct4_dog2,i,j)
            slice_matrix_1 = slice_mat_three(oct4_dog1,i,j)
            slice_matrix_3 = slice_mat_three(oct4_dog3,i,j)
            slice_matrix_4 = slice_mat_three(oct4_dog4,i,j)
            key_mat_1[i+1][j+1] = comp_matrix_1(slice_matrix_1,slice_matrix_2,slice_matrix_3)
            key_mat_2[i+1][j+1] = comp_matrix_1(slice_matrix_2,slice_matrix_3,slice_matrix_4)

    key_mat_final = key_mat_1 + key_mat_2

    img = cv2.imread('task2.jpg')

    img = img / 255
    print(key_mat_final.shape)
    for i in range(key_mat_final.shape[0]-2):
        for j in range(key_mat_final.shape[1]-2):
            if key_mat_final[i][j] >= 1:
                # We subtract the image by -9 because the keypoint matrix is padded but original image is not and multiply by 8 because the original octave by divided by 8.
                img[i*8-9][j*8-9][:] = 1

    cv2.namedWindow('Keypoints Octave4', cv2.WINDOW_NORMAL)
    cv2.imshow('Keypoints Octave4', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key_mat_final

#This function will create all octaves and DOGs.
def run_all():

    create_all_octave()
    DOG_1()
    DOG_2()
    DOG_3()
    DOG_4()

#This function will create all keypoints for octaves and DOGs previously generated using run_all(). Please make sure to run the function run_all() before this one.
def generate_keypoints():

    key_mat1 = keypoint_1()
    print(key_mat1.shape)
    key_mat2 = keypoint_2()
    print(key_mat2.shape)
    key_mat3 = keypoint_3()
    print(key_mat3.shape)
    key_mat4 = keypoint_4()
    print(key_mat4.shape)

    img = cv2.imread('task2.jpg')
    img = img/255
    #What we are doing below is projecting the keypoints into the original image. This was done for individual keypoints in aforementioned functions too.
    for i in range(key_mat1.shape[0]):
        for j in range(key_mat1.shape[1]):
            if key_mat1[i][j] >= 1:
                img[i - 1][j - 1][:] = 1
    for i in range(key_mat2.shape[0]):
        for j in range(key_mat2.shape[1]):
            if key_mat2[i][j] >= 1:
                img[i * 2 - 3][j * 2 - 3][:] = 1
    for i in range(key_mat3.shape[0]):
        for j in range(key_mat3.shape[1]):
            if key_mat3[i][j] >= 1:
                img[i * 4 - 5][j * 4 - 5][:] = 1
    for i in range(key_mat4.shape[0]):
        for j in range(key_mat4.shape[1]):
            if key_mat4[i][j] >= 1:
                img[i * 8 - 9][j * 8 - 9][:] = 1
    #print(img)
    cv2.namedWindow('Keypoints Original', cv2.WINDOW_NORMAL)
    cv2.imshow('Keypoints Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #We create a sample.txt file which has the coordinates of 2 column of the final keypoint matrix. The 5 left most value were selected by looking at coordinates for [1,1,1]
    f1 = open('sample.txt', 'w+')
    num = 0
    for i in range(1,img.shape[1]):
        for j in range(img.shape[0]):
            #if img[j][i][:] == 1:
            num = num + 1
            temp = str(img[j][i][:]) + ' ' + str(i) + ',' + str(j) + '\n'
            f1.write(temp)
            if num == img.shape[0]:
                break
        if num == img.shape[0]:
            break
    f1.close()

#run_all()
generate_keypoints()
