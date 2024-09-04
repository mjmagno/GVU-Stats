import requests
import os
import json
import cv2
import numpy as np


def download_champion_names(version):
    url = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        champion_data = data["data"]
        # Extract the champion names
        champion_names = [champion for champion in champion_data.keys()]
        
        return champion_names
    else:
        print("Failed to fetch champion.json")
        return []
def round_all_portraits(input_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Construct full file path
            file_path = os.path.join(input_folder, filename)
            # Load the image
            img = cv2.imread(file_path)
            # Check if the image is loaded properly
            if img is None:
                print(f"Failed to load image: {filename}")
                continue
            
            # Resize the image to the target size (width, height)
            resized_img = cv2.resize(img, (58,58), interpolation=cv2.INTER_AREA)

            
            # Convert the image to grayscale
            #gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            # Create a circular mask
            height, width = resized_img.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            radius = min(center[0], center[1], width - center[0], height - center[1])
            cv2.circle(mask, center, radius, 255, thickness=-1)
            
            # Apply the circular mask to the image
            rounded_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
            
            # Save the rounded image to the output directory
            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, rounded_img)
            print(f"Processed and saved: {filename}")
            
# Function to download champion portraits and create champion_templates dictionary
def download_champion_portraits(version):
    # The version of the champion images, you can get the latest version dynamically or use a known version
    base_url = f"http://ddragon.leagueoflegends.com/cdn/{version}/img/champion/"
    champions = download_champion_names(version)

    # Create a directory to store images if it doesn't exist
    os.makedirs("champion_portraits", exist_ok=True)

    for champion in champions:
        image_url = base_url + f"{champion}.png"
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(f"champion_portraits/{champion}.png", "wb") as file:
                file.write(response.content)
            print(f"Downloaded {champion}.png")
        else:
            print(f"Failed to download {champion}.png")
            
    round_all_portraits("champion_portraits","round_portraits")
def create_champion_template(version):
    champions = download_champion_names(version)
    champion_templates = {}
    for champion in champions:
        champion_templates[champion] = f"round_portraits/{champion}.png"
    return champion_templates
        

