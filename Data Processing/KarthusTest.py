import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
scoreboard_image = cv2.imread('SmallTest1.png')
karthus_template = cv2.imread('champion_portraits/Karthus.png')


# Display the original Karthus template
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(karthus_template, cv2.COLOR_BGR2RGB))
plt.title('Original Karthus Template')
plt.show()

# Resize the template if necessary (verify dimensions match the scoreboard's portraits)
# Assuming you want to resize to 58x58 (you might need to adjust this)
karthus_template_resized = cv2.resize(karthus_template, (54,54), interpolation=cv2.INTER_LANCZOS4)


# Display the resized Karthus template
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(karthus_template_resized, cv2.COLOR_BGR2RGB))
plt.title('Resized Karthus Template')
plt.show()

gray_scoreboard = cv2.cvtColor(scoreboard_image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(karthus_template_resized, cv2.COLOR_BGR2GRAY)

# Display the gray Karthus template
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(gray_template, cv2.COLOR_BGR2RGB))
plt.title('Resized Karthus Template')
plt.show()

# Perform template matching
result = cv2.matchTemplate(gray_scoreboard, gray_template, cv2.TM_CCORR_NORMED)

# Find the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Define the threshold for a valid match
threshold = 0.65  # Adjust this based on your needs
if max_val >= threshold:
    print(f"Karthus found with confidence: {max_val}")
    top_left = max_loc
    bottom_right = (top_left[0] + 58, top_left[1] + 58)
    cv2.rectangle(scoreboard_image, top_left, bottom_right, (0, 255, 0), 2)
else:
    print("Karthus not found or confidence too low.")

# Display the result
cv2.imshow("Matched Image", gray_scoreboard)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, display using matplotlib if cv2.imshow doesn't work well on your setup
plt.imshow(cv2.cvtColor(scoreboard_image, cv2.COLOR_BGR2RGB))
plt.title('Template Matching Result')
plt.show()