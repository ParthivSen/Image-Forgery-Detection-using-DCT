import numpy as np
from scipy import fft
import cv2
from operator import itemgetter


def read_img(img_path):
    image = img_path
    original_image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.imread(image, 0)

    overlay = original_image.copy()

    img = np.array(image)
    height, width = img.shape

    return img, original_image, overlay, width, height


def create_quantize_dct(img, width, height, block_size, stride, Q_8x8):
    quant_row_matrices = []

    for i in range(0, height - block_size, stride):
        for j in range(0, width - block_size, stride):
            block = img[i: i + block_size, j: j + block_size]

            dct_matrix = fft.dct(block)

            quant_block = np.round(np.divide(dct_matrix, Q_8x8))
            block_row = list(quant_block.flatten())

            quant_row_matrices.append([(i, j), block_row])

    return quant_row_matrices


def lexographic_sort(quant_row_matrices):
    sorted_blocks = sorted(quant_row_matrices, key=itemgetter(1))

    matched_blocks = []

    shift_vec_count = {}

    for i in range(len(sorted_blocks) - 1):
        if sorted_blocks[i][1] == sorted_blocks[i + 1][1]:
            point1 = sorted_blocks[i][0]
            point2 = sorted_blocks[i + 1][0]

            s = np.linalg.norm(np.array(point1) - np.array(point2))

            shift_vec_count[s] = shift_vec_count.get(s, 0) + 1
            matched_blocks.append([sorted_blocks[i][1], sorted_blocks[i + 1][1],
                                   point1, point2, s])

    return shift_vec_count, matched_blocks


def shift_vector_thresh(shift_vec_count, matched_blocks, shift_thresh):
    matched_pixels_start = []
    for sf in shift_vec_count:
        if shift_vec_count[sf] > shift_thresh:
            for row in matched_blocks:
                if sf == row[4]:
                    matched_pixels_start.append([row[2], row[3]])

    return matched_pixels_start


def display_results(overlay, original_image, matched_pixels_start, block_size):
    alpha = 0.5
    orig = original_image.copy()

    for starting_points in matched_pixels_start:
        p1 = starting_points[0]
        p2 = starting_points[1]

        overlay[p1[0]: p1[0] + block_size, p1[1]
            : p1[1] + block_size] = (0, 255, 0)
        overlay[p2[0]: p2[0] + block_size, p2[1]
            : p2[1] + block_size] = (0, 0, 255)

    cv2.addWeighted(overlay, alpha, original_image, 1, 0, original_image)

    cv2.imshow("Original Image", orig)
    cv2.imshow("Detected Forged/Duplicated Regions", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
