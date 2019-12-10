import os
from keras.applications import VGG16
from keras import models
from keras import layers

conv_base = VGG16(weights='imagenet', 
                    include_top=False, 
                    input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False


