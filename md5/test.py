import cv2
import hashlib
from skimage.metrics import structural_similarity as ssim


def calculate_md5(image_path):
    # Calculate MD5 hash of the image content
    md5 = hashlib.md5()
    with open(image_path, "rb") as image_file:
        while chunk := image_file.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def detect_image_forgery(original_image_path, suspect_image_path):
    # Calculate MD5 hashes for both images
    original_md5 = calculate_md5(original_image_path)
    suspect_md5 = calculate_md5(suspect_image_path)

    # Compare MD5 hashes
    if original_md5 == suspect_md5:
        print("MD5 Hashes Match: Image is not tampered.")
    else:
        print("MD5 Hashes Mismatch: Potential forgery detected.")

        # Load images using OpenCV
        original_image = cv2.imread(original_image_path)
        suspect_image = cv2.imread(suspect_image_path)

        # Perform basic content comparison (e.g., using Structural Similarity Index)
        # You may need to install scikit-image: pip install scikit-image

        # Convert images to grayscale for SSIM comparison
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        suspect_gray = cv2.cvtColor(suspect_image, cv2.COLOR_BGR2GRAY)

        # Compute SSIM score
        similarity_index = ssim(original_gray, suspect_gray)
        print(f"Structural Similarity Index (SSIM): {similarity_index}")


if __name__ == "__main__":
    original_image_path = "code\md5\original.jpg"
    suspect_image_path = "code\md5\suspect.jpg"

    detect_image_forgery(original_image_path, suspect_image_path)
