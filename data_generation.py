import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --- Configuration ---
OUTPUT_DIR = "data"
NUM_IMAGES = 20000  # Increased for a better dataset
IMG_SIZE = (256, 256)
CHAR_SET = string.ascii_uppercase + string.digits # Random A-Z and 0-9
MIN_CHARS = 3
MAX_CHARS = 7

# --- Font Setup ---
# Add paths to .ttf files on your system. 
# Windows: "C:/Windows/Fonts/arial.ttf" | Linux: "/usr/share/fonts/..."
FONT_PATHS = [
    "arial.ttf", 
    "verdanab.ttf", 
    "times.ttf", 
    "cour.ttf"
] 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_text():
    length = random.randint(MIN_CHARS, MAX_CHARS)
    return ''.join(random.choices(CHAR_SET, k=length))

def add_noise_and_lines(image):
    """Adds salt-and-pepper noise and random strike-through lines."""
    img_array = np.array(image)
    
    # Salt and Pepper Noise
    noise = np.random.randint(0, 255, (IMG_SIZE[0], IMG_SIZE[1]), dtype='uint8')
    img_array[noise < 3] = 0 
    img_array[noise > 252] = 255
    
    new_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(new_img)
    
    # Random Obstacle: A thin line crossing the text
    if random.random() > 0.5:
        x1, y1 = random.randint(0, 20), random.randint(40, 80)
        x2, y2 = random.randint(100, 128), random.randint(40, 80)
        draw.line((x1, y1, x2, y2), fill=random.randint(0, 100), width=1)
        
    return new_img

def generate_dataset():
    available_fonts = []
    for f in FONT_PATHS:
        try:
            available_fonts.append(ImageFont.truetype(f, random.randint(22, 32)))
        except:
            continue
    
    if not available_fonts:
        print("No custom fonts found, using default.")
        available_fonts = [ImageFont.load_default()]

    for i in range(NUM_IMAGES):
        # 1. Setup Canvas
        bg_color = random.randint(220, 255)
        img = Image.new('L', IMG_SIZE, color=bg_color)
        
        # 2. Random Text and Font
        text = get_random_text()
        font = random.choice(available_fonts)
        
        # 3. Render Text to Layer
        txt_layer = Image.new('L', IMG_SIZE, color=0)
        draw = ImageDraw.Draw(txt_layer)
        
        # Calculate centering
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((IMG_SIZE[0]-w)/2, (IMG_SIZE[1]-h)/2), text, fill=255, font=font)
        
        # 4. Random Rotation
        angle = random.uniform(-25, 25)
        rotated_txt = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # 5. Composite and Noise
        img.paste(ImageOps.colorize(rotated_txt, (0,0,0), (0,0,0)), (0,0), rotated_txt)
        img = add_noise_and_lines(img)
        
        # 6. Save with label in filename
        # Note: We replace any risky chars, though CHAR_SET here is safe.
        file_path = os.path.join(OUTPUT_DIR, f"{i}_{text}.png")
        img.save(file_path)

    print(f"Generated {NUM_IMAGES} images in /{OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
