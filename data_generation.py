import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUTPUT_DIR = "data"
NUM_IMAGES = 20000
IMG_SIZE = (256, 256)
CHAR_SET = string.ascii_uppercase + string.digits
MIN_CHARS = 3
MAX_CHARS = 7
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
    img_array = np.array(image)
    noise = np.random.randint(0, 255, (IMG_SIZE[0], IMG_SIZE[1]), dtype='uint8')
    img_array[noise < 3] = 0 
    img_array[noise > 252] = 255
    
    new_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(new_img)

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
        available_fonts = [ImageFont.load_default()]

    for i in range(NUM_IMAGES):
        bg_color = random.randint(220, 255)
        img = Image.new('L', IMG_SIZE, color=bg_color)
        
        text = get_random_text()
        font = random.choice(available_fonts)
        
        txt_layer = Image.new('L', IMG_SIZE, color=0)
        draw = ImageDraw.Draw(txt_layer)
        
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((IMG_SIZE[0]-w)/2, (IMG_SIZE[1]-h)/2), text, fill=255, font=font)
        
        angle = random.uniform(-25, 25)
        rotated_txt = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        img.paste(ImageOps.colorize(rotated_txt, (0,0,0), (0,0,0)), (0,0), rotated_txt)
        img = add_noise_and_lines(img)

        if os.path.exists(OUTPUT_DIR):
            pass
        else:
            os.mkdir(OUTPUT_DIR)
        file_path = os.path.join(OUTPUT_DIR, f"{i}_{text}.png")
        img.save(file_path)

    print(f"Generated {NUM_IMAGES} images in /{OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()

