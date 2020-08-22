from tensorflow.keras.models import load_model
from helpers import predict
import cv2
import numpy as np
from pathlib import Path

model = load_model(Path(__file__).resolve().parent / 'multi_digit_classification.h5')
image_path = Path(__file__).resolve().parent / 'geez.png'

def prep_predict(image_path):
    image = cv2.imread(str(image_path), 0)
    image = cv2.resize(image,(64,64))
    image = image.astype('float32')
    image /= 255
    return predict(image, model, 5)

#Predict Test set
print('\033[93m' + f'Your Answer is {prep_predict(image_path)}' '\033[0m')
#You can add your image path and run the script to predict or develop a terminal or 
#GUI to do that properly!