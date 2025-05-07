import os
import subprocess
import sys
# Check if gdown is installed
try:
    import gdown
except ImportError:
    print("Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def download(weights_dir, folder_id):
    os.makedirs(weights_dir, exist_ok=True)
    # Construct the gdown folder URL
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    subprocess.run(["gdown", "--folder", folder_url, "-O", weights_dir], check=True)

weights_dir = "./weights"
scorer_gdrive_id = "1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC"
refiner_gdrive_id = "12Te_3TELLes5cim1d7F7EBTwUSe7iRBj"
download(weights_dir, scorer_gdrive_id)
download(weights_dir, refiner_gdrive_id)
print("✅ Download complete.")
