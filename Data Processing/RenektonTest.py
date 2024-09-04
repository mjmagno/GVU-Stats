import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_and_create_mask(image_path, target_size):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Resize the image to the target size (width, height)
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    # Create a circular mask
    height, width = resized_img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(center[0], center[1], width - center[0], height - center[1])
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # Create an output image with transparency (4 channels)
    if resized_img.shape[2] == 3:  # If the image doesn't have an alpha channel, add one
        rounded_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2BGRA)
    else:
        rounded_img = resized_img.copy()

    # Apply the circular mask to each color channel
    for c in range(3):
        rounded_img[:, :, c] = cv2.bitwise_and(resized_img[:, :, c], mask)
    
    # Set the alpha channel to be the mask
    rounded_img[:, :, 3] = mask

      # Convert the rounded image back to BGR for template matching
    bgr_rounded_img = cv2.cvtColor(rounded_img, cv2.COLOR_BGRA2BGR)

    return bgr_rounded_img, mask

target_size = (55, 55)
# Load the images
scoreboard_image = cv2.imread('SmallTest1.png')
renekton_template, mask = resize_and_create_mask('champion_portraits/Renekton.png', target_size)


plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(renekton_template, cv2.COLOR_BGR2RGB))
plt.title('Rounded Renekton Template')
plt.show()



# Perform template matching
result = cv2.matchTemplate(scoreboard_image, renekton_template, cv2.TM_CCOEFF_NORMED, mask=mask)

# Find the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Define the threshold for a valid match
threshold = 0.6  # Adjust this based on your needs
if max_val >= threshold:
    print(f"Renekton found with confidence: {max_val}")
    top_left = max_loc
    bottom_right = (top_left[0] + 58, top_left[1] + 58)
    cv2.rectangle(scoreboard_image, top_left, bottom_right, (0, 255, 0), 2)
else:
    print("Renekton not found or confidence too low.")


# Optionally, display using matplotlib if cv2.imshow doesn't work well on your setup
plt.imshow(cv2.cvtColor(scoreboard_image, cv2.COLOR_BGR2RGB))
plt.title('Template Matching Result')
plt.show()