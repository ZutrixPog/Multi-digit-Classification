import numpy as np
import random
import cv2 #I know its a bit overkill
from tensorflow.keras.utils import to_categorical

#using raw mnist dataset to create a multi-digit numbers dataset
def generate_synth_data(data, labels, size):
    img_height = 64
    img_width = 64

    synth_data = np.ndarray((size, img_height, img_width))
    synth_labels = []

    for i in range(0,size):
        digits_number = random.randint(1,5)

        digits_indicies = [random.randint(0,len(data)-1) for n in range(0, digits_number)]

        new_image = cv2.hconcat([data[index] for index in digits_indicies])
        new_lable = [labels[index] for index in digits_indicies]

        for j in range(0, 5-digits_number):
            new_image = np.hstack([new_image, np.zeros((28,28))])
            new_lable.append(10)
        new_image = cv2.resize(new_image,(img_width, img_height))

        synth_data[i,...] = new_image

        synth_labels.append(tuple(new_lable))

    return synth_data, synth_labels

#convert labels to one-hot encoding
def to_mod_categorical(labels, classes=11):
    
    first_digit   = np.ndarray((len(labels),classes))
    seccond_digit = np.ndarray((len(labels),classes))
    third_digit   = np.ndarray((len(labels),classes))
    fourth_digit  = np.ndarray((len(labels),classes))
    fifth_digit   = np.ndarray((len(labels),classes))

    for i, label in enumerate(labels):

        first_digit[i, :] = to_categorical(label[0],classes)
        seccond_digit[i, :] = to_categorical(label[1],classes)
        third_digit[i, :] = to_categorical(label[2],classes)
        fourth_digit[i, :] = to_categorical(label[3],classes)
        fifth_digit[i, :] = to_categorical(label[4],classes)

    return [first_digit, seccond_digit, third_digit, fourth_digit, fifth_digit] 

#process image to be compatiable with our model
def process_images(images):
    images = images.reshape((len(images),64,64,1))
    images = images.astype('float32')
    images /= 255
    return images

#predict multiple images
def predict_classes(model, data, value_num, digits_num):
    preds = np.asarray(model.predict(process_images(data))) 
    predictions = []
    for i in range(0, value_num):
        pred = []
        for j in range(0,digits_num):
            pred.append(np.argmax(preds[i][j]))
        predictions.append(''.join(map(str,pred)))
    return predictions

#predict singel image
def predict(data,model, digits):
    data = model.predict(data.reshape((1,64,64,1)))
    result = []
    for j in range(digits):
        result.append(np.argmax(data[j]))
    return ''.join(map(str,result))  
