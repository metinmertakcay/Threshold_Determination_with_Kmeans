# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:22:31 2019
@author: Metin Mert Akçay
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

FOLDER_PATH = "bird images"
MAX_ITERATION = 20

""" 
    This function is used to read the image as grayscale.
    @param image_name: name of the image to be read
    @return grayscale image
"""
def read_image_as_grayscale(image_name):
    return cv2.imread(os.path.join(FOLDER_PATH, image_name), cv2.IMREAD_GRAYSCALE)


""" 
    This function is used to perform convolution operation.
    @param image: matrix to be convolution operation
    @param kernel: matrix to be used in convolution
    @return img: matrix formed by convolution
"""
def convolution(image, kernel):
    image_row_len = len(image)
    image_col_len = len(image[0])
    i = int(len(kernel) / 2)
    
    # Created new matrix for convolution operation
    img = []
    while (i < image_row_len - int(len(kernel) / 2)):
        j = int(len(kernel) / 2)
        new_image_col = []
        while (j < image_col_len - int(len(kernel) / 2)):
            pixel_value = 0
            k = (-1) * int(len(kernel) / 2)
            while (k <= int(len(kernel) / 2)):
                l = (-1) * int(len(kernel) / 2)
                while (l <= int(len(kernel) / 2)):
                    pixel_value += image[i + k][j + l] * kernel[k + int(len(kernel) / 2)][l + int(len(kernel) / 2)]
                    l +=1
                k += 1
            new_image_col.append(int(round(pixel_value)))
            j += 1
        img.append(new_image_col)
        i += 1
    return img


""" 
    This function is used for blur the image to remove noise.
    @param image: grayscale image
    @return: blurred image
"""
def gauss(image): 
    gauss_kernel = np.array([[1, 4,  7,  4,  1],
                             [4, 16, 26, 16, 4],
                             [7, 26, 41, 26, 7],
                             [4, 16, 26, 16, 4],
                             [1, 4,  7,  4,  1]]) / 273;
    return np.asarray(convolution(image, gauss_kernel))


"""
    This function is used for the collection of elements in the list
    @param cluster_list: A list which include pixel values
    @return total: Sum of elements in the list
"""
def summation(cluster_list):
    total = 0
    for i in range(len(cluster_list)):
        total += int(cluster_list[i])
    return total


"""
    This function is used to calculate kmeans error
    @param xc1: first cluster center current pixel value
    @param xc2: second cluster center current pixel value
    @param xp1: first cluster center previous pixel value
    @param xp2: second cluster center previous pixel value
    @return calculated error
"""
def calculate_error(xc1, xc2, xp1, xp2):
    return pow(pow((xc1 - xp1), 2), 0.5) + pow(pow((xc2 - xp2), 2), 0.5)


"""
    This function is used to find threshold value for image segmentation using kmeans algoritm.
    @param image: image to be applied kmeans to pixels.
    @return threshold value
"""
def kmeans(image):
    row = image.shape[0]
    column = image.shape[1]
    
    # the first one is the current center(pixel value), the second one is the previous center(pixel value)
    # It is predicted that there will be 2 clusters. The first cluster represents background, the second cluster represents foreground.
    first_cluster_center = [0, 255]
    second_cluster_center = [255, 0]

    iteration = 0
    while iteration < MAX_ITERATION and calculate_error(first_cluster_center[0], second_cluster_center[0], first_cluster_center[1], second_cluster_center[1]) > 0.00001:
        first_cluster = []
        second_cluster = []
        for i in range(row):
            for j in range(column):
                if abs(float(image[i][j]) - first_cluster_center[0]) <= abs(float(image[i][j]) - second_cluster_center[0]):
                    first_cluster.append(int(image[i][j]))
                else:
                    second_cluster.append(int(image[i][j]))
        # find new cluster center pixel value.
        first_cluster_center = [summation(first_cluster) / len(first_cluster), first_cluster_center[0]]
        second_cluster_center = [summation(second_cluster) / len(second_cluster), second_cluster_center[0]]
        iteration += 1
    # The line separating the clusters is determined as the threshold value.
    return (max(first_cluster) + min(second_cluster)) / 2


"""
    This function is used to apply threshold value on image.
    @param image: image to apply threshold
    @param thresh: threshold value
"""
def apply_threshold(image, thresh):
    row = image.shape[0]
    column = image.shape[1]
    
    for i in range(row):
        for j in range(column):
            # The pixel value lower than the threshold value is specified as the background and the higher values ​​are specified as the foreground.
            if image[i][j] < thresh:
                image[i][j] = 255
            else:
                image[i][j] = 0


"""
    This function is used to do erosion operation
    @param image: image to be process
    @param img: erosion operation result
"""
def erosion(image):
    row = image.shape[0]
    column = image.shape[1]
    
    img = image.copy()
    for i in range(row):
        for j in range(column):
            if image[i][j] == 255:
                if i + 1 >= row or j + 1 >= column or i - 1 < 0 or j - 1 < 0:
                    img[i][j] = 0
                elif image[i][j + 1] != 255 or image[i][j - 1] != 255 or image[i + 1][j] != 255 or image[i - 1][j] != 255:
                    img[i][j] = 0
    return img


""" 
    This function is used for show image
    @param image: image to show
    @param image_name: name of the image
"""
def show_image(image, image_name):
    plt.imshow(np.array(image), cmap='gray', vmin=0, vmax=255)
    plt.title(image_name)
    plt.show()


"""
    This function is used to fill area which is connected each other with same label
    @param image: image
    @param i: row index
    @param j: column index
    @param label: pixel value will change 0 to label to prevent count again same bird
"""
def fill_with_same_label(image, i, j, label):
    # 8 neighbours
    nbx_8 = [-1, -1, -1, 0, 0, 1, 1, 1]
    nby_8 = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    image[i][j] = label
    for x, y in zip(nbx_8, nby_8):
        if image[i + x][j + y] == 255:
            fill_with_same_label(image, i + x, j + y, label)


"""
    This function is used to count birds
    @param image: image
    @return label: number of bird
"""
def connected_component_labeling(image):
    label = 0
    row = image.shape[0]
    column = image.shape[1]
    
    for i in range(row):
        for j in range(column):
            if image[i][j] == 255:
                label += 1
                fill_with_same_label(image, i, j, label)
    return label


""" This is where the code starts """
if __name__ == '__main__':
    # Get image names given folder
    image_list = os.listdir(FOLDER_PATH)

    for image_name in image_list:
        image = read_image_as_grayscale(image_name)
        image = gauss(image)
        thresh = kmeans(image)
        apply_threshold(image, thresh)
        image = erosion(image)
        show_image(image, image_name)
        bird_count = connected_component_labeling(image)
        print("Bird count --> ", bird_count)
        