import argparse
from quant_matrix import QuantizationMatrix
from utils.helper_utils import (
    read_img, create_quantize_dct, lexographic_sort, shift_vector_thresh, display_results
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--img', required=True,
                        help="Path of the image on which operation needs to be performed")
    parser.add_argument('--block_size', type=int, default=8, help="Block Size")
    parser.add_argument('--qf', type=float, default=0.75,
                        help="Quality Factor")
    parser.add_argument('--shift_thresh', type=int, default=10,
                        help="Threshold for shift vector count")
    parser.add_argument('--stride', type=int, default=1,
                        help="Sliding window stride count / overlap")

    args = parser.parse_args()

    img_path = args.img
    block_size = args.block_size
    qf = args.qf
    shift_thresh = args.shift_thresh
    stride = args.stride

    Q_8x8 = QuantizationMatrix().get_qm(qf)

    img, original_image, overlay, width, height = read_img(img_path)

    quant_row_matrices = create_quantize_dct(
        img, width, height, block_size, stride, Q_8x8)

    shift_vec_count, matched_blocks = lexographic_sort(quant_row_matrices)

    matched_pixels_start = shift_vector_thresh(
        shift_vec_count, matched_blocks, shift_thresh)

    display_results(overlay, original_image, matched_pixels_start, block_size)
