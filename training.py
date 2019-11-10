import sys
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

arguments=sys.argv


#number of pixels to which each image will be reduced before the training process
IMG_W = 200   
IMG_H = 250

#setting up the directory variables for training.
if(len(arguments)==1):
	train_dir=os.getcwd()
else :
	train_dir=os.getcwd()+'/'+arguments[1]
	

#sets up the image batch augmentation :
	#creates new images randomly modified by tranformations, zoom or flips based on the parameters given
	#this step helps create more images for fitting the model if too few images have been provided for training
train_datagen = keras.preprocessing.image.ImageDataGenerator(
				rescale=1./255,       #rescale bitmap values from 0 - 255  to  0 - 1
				rotation_range=30,
				width_shift_range=0.1,
				height_shift_range=0.1,
				zoom_range=0.3,
				horizontal_flip=True)
	
	#flows images from the subdirectories of "train_dir", associating classes based on subfolders within the directory passed as argument.
train_generator = train_datagen.flow_from_directory(
				train_dir,
				target_size=(IMG_W, IMG_H),  #rescale images from 2462 x 2056 to 250 x 200
				class_mode='binary')


#construction of the ANN model, consisting of 2 convolutional layers, 2 max pooling layers and 2 dense perceptron layers
#Convolutional layers create filters that "learn" to identify features in images
#MaxPooling layers reduce the size of the matrix by associating the highest value of a group of pixels as the new value of a single pixel, 
# also enhancing features of the image and reducing computational time.
#Dense layers are composed of multiple perceptrons fully connected to the previous layer.
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_W, IMG_H, 3)), 	
	tf.keras.layers.MaxPooling2D(2, 2),			

	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),

	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(64, activation='relu'),   #rectified linear units activation function - no negative values
	tf.keras.layers.Dense(64, activation='tanh'), 	#hiperbolic tangent activation function - values ranging from -1 to 1
	tf.keras.layers.Dense(2, activation='softmax')  #softmax activation function - outputs are arrays of 0s or 1s based on classes
])

#Compiles the model, associating learning rate optimization functions, error and metrics for training session
model.compile(
		optimizer='rmsprop',     #widely used optimization function, gives better results in training session. adam is another good option
		loss='sparse_categorical_crossentropy',  #
		metrics=['accuracy'])

#Sets up the Early Stopping callback, that stops the training session if the loss metric stops decreasing after a set number of epochs.
es = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

#Begins the training session using the train data generator for a set maximum number of epochs.
start_time = time.time()
hist = model.fit_generator( 
		train_generator,
		steps_per_epoch=100,   #number of train examples fed to the model for each epoch, generated randomly using keras preprocessing.
		epochs=50,
		callbacks=[es])

elapsed= time.time() - start_time
print("Elapsed time: %.10f s"%(elapsed))

model.save(os.getcwd()+'/saved_model.h5')

print("done! -- run evaluation.py to evaluate the model or classify new images")
os.system("echo -n '\a';sleep 0.2;"*2)

