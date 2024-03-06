import cv2
import numpy as np


def calculate_dcp(image):
    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Extract the L channel (luminance)
    l_channel = lab_image[:, :, 0]

    # Compute the mean and standard deviation of the L channel
    mean_l, std_l = np.mean(l_channel), np.std(l_channel)

    # Threshold for dense color patches
    threshold = mean_l + 2 * std_l

    # Create a binary mask for dense color patches
    dense_patch_mask = (l_channel > threshold).astype(np.uint8)

    return dense_patch_mask


def detect_image_forgery(original_image_path, suspect_image_path):
    # Load images using OpenCV
    original_image = cv2.imread(original_image_path)
    suspect_image = cv2.imread(suspect_image_path)

    # Calculate DCP for both images
    original_dcp = calculate_dcp(original_image)
    suspect_dcp = calculate_dcp(suspect_image)

    # Perform a pixel-wise comparison
    difference_map = cv2.absdiff(original_dcp, suspect_dcp)

    # Count the number of non-zero pixels in the difference map
    non_zero_pixels = np.count_nonzero(difference_map)

    # Define a threshold for forgery detection
    threshold = 0.02 * original_image.size

    if non_zero_pixels > threshold:
        print("Forgery Detected: Images differ significantly in dense color patches.")
    else:
        print("Images appear authentic: No significant differences in dense color patches.")


if __name__ == "__main__":
    original_image_path = "code\md5\original.jpg"
    suspect_image_path = "code\md5\suspect.jpg"

    detect_image_forgery(original_image_path, suspect_image_path)
