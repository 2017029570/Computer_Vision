import numpy as np
import cv2 as cv


def Harris(img, window_size):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobel_x = cv.Sobel(img_gray, cv.CV_32F, 1, 0)
    sobel_y = cv.Sobel(img_gray, cv.CV_32F, 0, 1)

    A = sobel_x * sobel_x
    B = sobel_x * sobel_y
    C = sobel_y * sobel_y

    offset = window_size//2

    height, width = img_gray.shape
    harris = np.zeros((height, width))
    
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_A = np.sum(A[y-offset:y+offset+1, x-offset:x+offset+1])
            window_B = np.sum(B[y-offset:y+offset+1, x-offset:x+offset+1])
            window_C = np.sum(C[y-offset:y+offset+1, x-offset:x+offset+1])

            det = (window_A * window_C) - (window_B ** 2)
            trace = window_A + window_C

            R = det - 0.03 * (trace ** 2)
            harris[y, x] = R

    max_val = harris.max()
    limit= 0.1 * max_val

    z = np.zeros((max(harris.shape), max(harris.shape)))
    z[:harris.shape[0], :harris.shape[1]] = harris
    harris = z

    w, v = np.linalg.eig(harris)

    print(min(w) ** (-1/2), max(w) ** (-1/2))
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if harris[y, x] > limit:
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 0)
                img.itemset((y, x, 2), 255)
                
    return img

img = cv.imread('checkerboard.png')
newImg = cv.imwrite("Harris_3.jpg", Harris(img, 3))
newImg = cv.imwrite("Harris_7.jpg", Harris(img,7))
newImg = cv.imwrite("Harris_11.jpg", Harris(img, 11))



