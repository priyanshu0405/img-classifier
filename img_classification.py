# Image Classifier using CNN

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
plt.style.use('fivethirtyeight')

#defining the categories
category = ["Alipore", "Balurghat", "Bankura", "Barasat", "Bardhaman", "Berhampore", "Chinsurah", "Cooch Behar", "Darjeeling", "Englishbajar", "Hawrah", "Jalpaiguri", "Kolkata", "Krishnanagar", "Malda", "Medinipur", "Purulia", "Raiganj", "Suri", "Tomluk"]

datadir = "/content/drive/My Drive/dataset"

train_data = []

#converting image dataset to pixel array and appending to train_data
for c in category:
  path = os.path.join(datadir,c)
  for img in os.listdir(path):
    data_array = cv2.imread(os.path.join(path,img))
    new_array = cv2.resize(data_array, (100,60))  #resizing all images to same size
    train_data.append([new_array])

#making the label array
import array
label_data = array.array('i')
for i in range(1,8001):
  label_data.append(x)
  if i%400 == 0:
    x+=1

label_one_hot = to_categorical(label_data)


#making the model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(60,100,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(20, activation='softmax'))   #20 argument is given for 20 classes that we have defined
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


#normalizing the pixels
train_data = np.array(train_data)
train_data = train_data / 255
train_data.shape

train_data = np.squeeze(train_data)


#traing the model
model.fit(train_data,label_one_hot, epochs=100)



datadir_2 = "/content/drive/My Drive/train"

test_data = []

#making the test data
for c in category:
  path = os.path.join(datadir_2,c)
  for img in os.listdir(path):
    data_array_2 = cv2.imread(os.path.join(path,img))
    new_array_2 = cv2.resize(data_array_2, (100,60))
    test_data.append([new_array_2])


#label array for test data
import array
label_data_2 = array.array('i')
x=int(0)
for i in range(1,2001):
  label_data_2.append(x)
  if i%100 == 0:
    x+=1

test_data = np.array(test_data)
test_data = test_data / 255
test_data.shape

test_data = np.squeeze(test_data)

label_one_hot_2 = to_categorical(label_data_2)


#testing the model
model.evaluate(test_data,label_one_hot_2)


