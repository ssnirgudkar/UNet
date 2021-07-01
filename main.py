import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from UNet_model import create_model 
from CreateImageDirPaths import createImageDirPaths 
from DataSequence import MarineImages 
import PIL

import random

imageSize = (640, 512)
numClasses = 7
# sky:0, water:1, structure:2, Obstacle:3, living obstacle:4, background:5, self:6
batchSize = 32 # This may not be possible.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

[input_img_paths, segmentation_img_paths] = createImageDirPaths()

print("input img paths = {0}".format(input_img_paths))
print("Type of input_img_paths = {0}".format(type(input_img_paths)))


# Build model
model = create_model(imageSize, numClasses)
model.summary()

# split image path into a training and a validation set
validation_samples = 5
random.Random(15).shuffle(input_img_paths)
random.Random(15).shuffle(segmentation_img_paths)
train_input_img_paths = input_img_paths[:-validation_samples]
train_segmentation_img_paths = segmentation_img_paths[:-validation_samples]
validation_input_img_paths = input_img_paths[-validation_samples:]
validation_target_img_paths = segmentation_img_paths[-validation_samples:]

marineData_training = MarineImages(batchSize, imageSize, train_input_img_paths, train_segmentation_img_paths)
marineData_validation = MarineImages(batchSize, imageSize, validation_input_img_paths, validation_target_img_paths)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callback = [tf.keras.callbacks.ModelCheckpoint("uml_segmentation.h5", save_best_only=True)]

epochs = 10
model.fit(marineData_training, epochs=epochs, validation_data=marineData_validation, callbacks=callback)