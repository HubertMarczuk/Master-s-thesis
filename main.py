from PIL import Image

def preprocess_image(input_path):
    img = Image.open(input_path)
    img = img.convert("RGB")
    
    img.thumbnail((9999, 256), Image.LANCZOS)
    
    width, height = img.size
    left = (width - 256)/2
    top = (height - 256)/2
    right = (width + 256)/2
    bottom = (height + 256)/2
    img = img.crop((left, top, right, bottom))
    
    img.show()

path = "Master-s-thesis/photos/Audi/IMG_20250311_104415.jpg"
preprocess_image(path)