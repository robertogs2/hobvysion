import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

np.set_printoptions(suppress=True)

# Create a sample image for demonstration
w, h = 300, 300
image = np.zeros((w, h, 3), dtype=np.uint8)
cv2.putText(
    image,
    "OpenCV",
    (50, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (255, 255, 255),
    5,
    cv2.LINE_AA,
)

# Define an intrinsic camera matrix
focal_length = (
    w  # Focal length is set to the width of the image for better visualization
)
cx, cy = w / 2, h / 2  # Principal point (center of the image)
K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

# Steps:
# 1. Move image to origin (almost manually, matrix is dumb with -cx, -cy)
# 2. Rotate that in the origin
# 3. Move it backwards to focal plane
# 4. Project those points to what the camera will see

# Matrix of original points (corners of the image)
corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).T
corners_3d_homogeneous = np.vstack(
    (corners, np.zeros((1, corners.shape[1])), np.ones((1, corners.shape[1])))
)  # Convert to 3D and homogeneous coordinates


# Warps around the Y-axis, which is the vertical axis in the image
def warp_image(angle_degrees):
    # Translate points to the origin

    T = np.array(  # noqa: N806
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    corners_translated = T @ corners_3d_homogeneous

    # Rotate around the Y-axis
    angle = np.radians(angle_degrees)  # Rotation angle in radians
    R = np.array(  # noqa: N806
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )

    # Also push the points back to the focal plane after rotation
    t = np.array([[0, 0, focal_length]]).T

    # Combine the rotation and translation into the "extrinsic" matrix
    extrinsics = np.hstack((R, t))
    corners_translated_rotated = extrinsics @ corners_translated

    # Project the points using the intrinsic camera matrix
    corners_projected = K @ corners_translated_rotated
    corners_projected /= corners_projected[2, :]  # Normalize by the third coordinate
    corners_projected = corners_projected[:2, :]  # Keep only the x and y coordinates

    # Get the perspective transformation matrix
    m = cv2.getPerspectiveTransform(
        np.float32(corners.T), np.float32(corners_projected.T)
    )

    # Apply the perspective warp transformation. cv2.warpPerspective uses backward mapping already.
    warped_image = cv2.warpPerspective(image, m, (w, h))

    return warped_image


# Initial angle
init_angle = 0

# Create the figure and the axes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Display the original image
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[0].axis("off")

# # Display the warped image
warped_image = warp_image(init_angle)
im_warped = axs[1].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
axs[1].set_title("Warped Image")
axs[1].axis("off")

# Define the slider axis and create the slider
ax_angle = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="lightgoldenrodyellow")
slider_angle = Slider(ax_angle, "Angle", -180, 180, valinit=init_angle)


# Update function for the slider
def update():
    angle = slider_angle.val
    warped_image = warp_image(angle)
    im_warped.set_data(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()


# Attach the update function to the slider
slider_angle.on_changed(update)

plt.show()
