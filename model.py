import numpy as np 
import pandas as pd
import random 
import matplotlib.pyplot as plt 
from helpers import generate_synth_data, to_mod_categorical, process_images, predict_classes, predict
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

#loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#generating our multi-digit dataset using the mnist dataset 
x_mod_train, y_mod_train = generate_synth_data(x_train, y_train, 60000)
x_mod_test, y_mod_test = generate_synth_data(x_test, y_test, 10000)

#convert to one-hot encoding
y_train = to_mod_categorical(y_mod_train)
y_test  = to_mod_categorical(y_mod_test)

#process images to be campatible with our model
x_mod_train = process_images(x_mod_train)
x_mod_test  = process_images(x_mod_test)

#defining the model using Functional Api
inputs = Input((64,64,1))

layers    = Conv2D(32, (4,4), (1,1), input_shape=(64,64,1), activation='relu')(inputs)
layers    = MaxPooling2D(pool_size=(2,2))(layers)
layers    = Conv2D(32, (4,4), (1,1), activation='relu')(layers)
layers    = MaxPooling2D(pool_size=(2,2))(layers)
layer_out = Flatten()(layers)
layers2   = Dense(128, activation='relu')(layer_out)
model     = Dropout(0.4)(layers2)
c0 = Dense(11, activation='softmax')(layers2)
c1 = Dense(11, activation='softmax')(layers2)
c2 = Dense(11, activation='softmax')(layers2)
c3 = Dense(11, activation='softmax')(layers2)
c4 = Dense(11, activation='softmax')(layers2)

model = Model(inputs=inputs, outputs=[c0, c1, c2, c3, c4])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', patience=2)
log_directory = 'logs\\fit'
board = TensorBoard(log_dir=histogram_freq=1,write_graph=True, wirte_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1)

#training the model
model.fit(x_mod_train, y_train, epochs=30, verbose=1, callbacks=[early, board], validation_data=(x_mod_test, y_test), use_multiprocessing=True)


print('Model Evaluation----------------------------------------')
eval = pd.DataFrame(model.history.history)
eval[['loss', 'val_loss']].plot()
print(model.evaluate(x_mod_test, y_test, verbose=1))
print('Test set Results----------------------------------------')
print(predict_classes(model, x_mod_test, 10000, 5))

if __name__ == '__main__':
    model.save('multi_digit_classification.h5')