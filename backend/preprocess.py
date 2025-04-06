import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path, img_size=(128, 128)):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
