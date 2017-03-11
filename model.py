

import csv
import os

def load_log():
    """Returns the training set, x_train and y_train, from the specified csv file"""
    rows = []
    with open('./sim/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    return rows

samples = load_log()

print(len(samples))


import matplotlib.pyplot as plt
import cv2

def get_image(sample):
    source_path = sample
    token = source_path.split('/')
    fn = token[-1]    
    local_path = './sim/IMG/'+fn
    img=cv2.imread(local_path)
    return img


# img = cv2.cvtColor(get_image(samples[0][0]), cv2.COLOR_BGR2RGB)
# 
# %matplotlib inline
# plt.imshow(img)


def augment_dataset(images, measurements):    
    """augments the training data with mirrored images. Returns a tuple"""
    aug_images = []
    aug_measurements = []
    
    for img, mes in zip(images, measurements):
        aug_images.append(img)
        aug_measurements.append(mes)
        
        flipped_img = cv2.flip(img,1)
        flipped_mes = mes * -1.0
        
        aug_images.append(flipped_img)
        aug_measurements.append(flipped_mes)
    
    return np.array(aug_images), np.array(aug_measurements)


# In[18]:

import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


def generator(samples, batch_size):
    images = []
    measurements = []    
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        num_samples = len(samples)
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            for row in batch_samples:
                #0: center 1: left 2:right image
                for column in range(3): # loops through the first 3 columns, 
                    images.append(get_image(row[column]))

                #4th (3) column in excel, measure corr. to center image
                measurement = float(row[3]) #4th column in the table
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)

                x_train, y_train = augment_dataset(np.array(images), np.array(measurements))      

            yield x_train, y_train


from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



import numpy as np
my_shape = (160, 320 ,3)



import keras
keras.backend.image_dim_ordering()

#my_shape= x_train.shape[1:4]
print(my_shape)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, MaxPooling2D, AveragePooling2D, Cropping2D, Dropout

#architecture of the model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape = my_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(Dropout(30))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(1))
#

print('len', len(train_samples))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)

model.save('model.h5')
