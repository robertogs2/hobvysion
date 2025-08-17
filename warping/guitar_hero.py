# This script demonstrates how to warp an image using OpenCV.
# It allows you to adjust the warp parameters using sliders.
# The warping simulates a perspective transformation based on user-defined parameters to give a Guiter Hero style.

import cv2
import numpy as np

# Load your source image
src = cv2.imread("test.png")
h, w = src.shape[:2]


# Callback function for trackbars (does nothing, we read values directly)
def nothing(x):
    pass


# Create window and sliders
cv2.namedWindow("Warp")
cv2.createTrackbar("a", "Warp", 0, 100, nothing)  # a from 0.0 to 1.0
cv2.createTrackbar("b", "Warp", 0, 100, nothing)  # b from 0.0 to 1.0

while True:
    # Read slider values and normalize to 0-1
    a = cv2.getTrackbarPos("a", "Warp") / 100.0
    b = cv2.getTrackbarPos("b", "Warp") / 100.0

    # Source points (corners of the image)
    src_pts = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])

    # Target points based on a,b
    dst_pts = np.float32(
        [
            [a, 0],  # (0,0) -> (a,0)
            [1 - a, 0],  # (1,0) -> (1-a,0)
            [0, 1 - b],  # (0,1) -> (0,1-b)
            [1, 1 - b],  # (1,1) -> (1,1-b)
        ]
    )

    # Compute homography using normalized points
    H_n, _ = cv2.findHomography(src_pts, dst_pts)
    print(f"a: {a}, b: {b}")
    print("Normalized Homography matrix:\n", H_n)

    # Since we computed the homography with all points normalized
    # We can't use the homography over the whole image directly, we need to normalize the points first
    # So multiply the Homography by a normalization matrix
    N = np.array(
        [
            [1 / w, 0, 0],
            [0, 1 / h, 0],
            [0, 0, 1],
        ]
    )
    H = np.linalg.inv(N) @ H_n @ N

    # Apply warp
    warped = cv2.warpPerspective(src, H, (w, h))

    # Show
    cv2.imshow("Warp", warped)

    # Exit on ESC
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
