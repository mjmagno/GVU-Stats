# Given a screen shot of League of Legends match data
# Match champion portraits to champion names, summoner name, and match stats

import cv2
import pytesseract
from PIL import Image
import pandas as pd
from download_champion_portraits import create_champion_template,download_champion_portraits
import time

# Function to load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    #img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    return img

# Function to perform template matching
def match_champion_portraits(scoreboard_image, champion_templates):
    matched_champions = {}
    for champion, template_path in champion_templates.items():
        
        template = preprocess_image(template_path)
        result = cv2.matchTemplate(scoreboard_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = 0.8  # You can adjust the threshold for matching accuracy
        if max_val >= threshold:
            top_left = max_loc
            matched_champions[top_left] = champion
    return matched_champions
# Function to extract text from image using OCR
def extract_text(image_path):
    img = preprocess_image(image_path)
    return pytesseract.image_to_string(img)



scoreboard_image_path = "SmallTest1.png"

# Load and preprocess the scoreboard image
scoreboard_image = preprocess_image(scoreboard_image_path)
champion_templates = create_champion_template("14.17.1")
# Perform template matching to identify champions
matched_champions = match_champion_portraits(scoreboard_image, champion_templates)
print(matched_champions)
# Extract text from the image using OCR
#scoreboard_text = extract_text(scoreboard_image_path)
#print(scoreboard_text)

