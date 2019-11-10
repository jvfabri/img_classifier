import sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

IMG_W = 200
IMG_H = 250

#Loads the previosly trained model for evaluation and prediction
model=keras.models.load_model('saved_model.h5')

#Evaluates the model's sparse categorical crossentropy, accuracy and response time based on images from the test folder.
#Evaluation is skipped if "--notest" flag is passed as argument in when executing the script.
if( not ('--notest' in sys.argv) ):
	#Loads the images to be used for evaluation of the model using a keras image preprocessing generator
	eval_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	val_gnrtr = eval_datagen.flow_from_directory('test',  target_size=(IMG_W, IMG_H), class_mode='binary')	

	start_time= time.time() #start computation of elapsed time
	output=model.evaluate_generator(val_gnrtr,21)
	elapsed= time.time() - start_time
	print("SCC (Loss): %.10E, Accuracy: %.3f, Elapsed Time: %.10f s"%(output[0],output[1],elapsed))


#Classifies a given .bmp file using the artificial neural network, printing on terminal the class which it belongs to.
#The file path is passed as a argument in terminal. The path needs to end in a file name that exists,
# or else a compilation error will be thrown.
if(len(sys.argv)>1 and ('.bmp' in sys.argv[1])):
	img = keras.preprocessing.image.load_img(sys.argv[1], target_size=(IMG_W, IMG_H)) #opens the file passed as argument and resizes
	img = keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0) 
	print('Sem anel' if model.predict_classes(img)==1 else 'Com anel') #shows in terminal the result of classification


	model = keras.models.Model(inputs=model.inputs, outputs=model.layers[2].output)

	feature_maps = model.predict(img)
	spc=0.1
	i = 1
	plt.figure(figsize=(6,6))
	for _ in range(4):
		for _ in range(4):
			ax = plt.subplot(4, 4, i)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(feature_maps[0, :, :, i-1], cmap='gray')
			i += 1
	plt.subplots_adjust(wspace = spc,hspace =spc)
	plt.savefig("features_"+sys.argv[1].replace('/','_').replace('.','_')+".pdf")
	plt.show()

