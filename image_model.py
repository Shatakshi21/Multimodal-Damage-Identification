# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:56:39 2019

@author: DELL
"""

import numpy as np
import keras
from keras.layers import Dense, Input, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Embedding,Activation,Flatten
from keras.models import Model,Sequential
#from keras.initializers import Constant
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing.image import ImageDataGenerator


train_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_image/train'
val_path='C:/Users/DELL/Desktop/Classification/new_multimodal/new_image/val'
img_height=299
img_width=299
train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=64,
    class_mode='categorical')
#x, y = train_generator.next()
#print(list(x))
#y_arg= y.argmax(axis=-1)
val_generator=val_datagen.flow_from_directory(val_path,
                                            target_size = (img_height,img_width),
                                            batch_size =64,shuffle=False,
                                            class_mode = 'categorical')
#y_true = val_generator.classes
#y_pred = numpy.rint(predictions) or predict_generator
#bulld the classifier------------------------------------------------------------------------
#model=keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=max, classes=6)
from keras.layers import Input
input_img = Input(shape = (299,299, 3))
from keras.layers import Conv2D, MaxPooling2D
tower_1 = Conv2D(256, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(256, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
#output=keras.layers.GlobalMaxPooling2D()(output)
output=Dense(256, activation='relu')(output)
from keras.layers import Flatten, Dense
img_out = Flatten()(output)
out    = Dense(6, activation='softmax')(img_out)
from keras.models import Model
model = Model(inputs = input_img, outputs = out)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_generator,
                         steps_per_epoch= 4663/64,
                         nb_epoch = 40,
                         validation_data = val_generator,
                          validation_steps= 1168)
model.save('image_classification.h5')
#---------------------test-----------------------
from keras.models import load_model
image_model=load_model('image_classification.h5')
loss,accuracy=image_model.evaluate_generator(val_generator)
print('image model loss:',loss)
print('image model accuracy:',accuracy)



