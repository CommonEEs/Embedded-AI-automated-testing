import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
import h5py as h5
import random
from tensorflow.keras import layers
from keras.layers.core import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#import dataset

import pathlib

dataset_url = "/home/root/Vitis-AI/examples/VART/resnet50_tensorflow/Dataset"
data_dir = pathlib.Path(dataset_url)


#split the dataset
img_height,img_width=299,299
batch_size=32
#produce the training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#produce the validation data  
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  

class_names = train_ds.class_names
print(class_names)
#visualize the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
plt.close()


#import pretrained model
resnet_model = Sequential()

#pretrained_model= tf.keras.applications.ResNet50(include_top=False,
#                   input_shape=(180,180,3),
#                   pooling='avg',classes=5,
#                   weights='imagenet')

#filepath = "/home/joncon/resnet50_tensorflow/model.h5"
pretrained_model = tf.keras.models.load_model(
			"/home/root/Vitis-AI/examples/VART/resnet50_tensorflow/NN.h5",
			custom_objects=None, 
			compile=True, 
			options=None
			)


for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)


#finalize
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(2, activation='softmax'))
resnet_model.summary()


#compile the model
resnet_model.compile(optimizer=Adam(learning_rate=0.001),
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
			)
history = resnet_model.fit(train_ds, 
			validation_data=val_ds, 
			epochs=10
			)


#model evaluation for accurracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

#model evaluation for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

#model testing
import cv2
#testdatadir = pathlib.Path("/home/joncon/SPII_Files/")
#testdir = list(testdatadir.glob('Testing Dataset/*.jpg'))

testdatadir = pathlib.Path("/home/root/Vitis-AI/examples/VART/")
testdir = list(testdatadir.glob('Full Dataset/*.jpg'))


for c in range(10):
    import cv2          #added to see if it fixes the error 
	t = random.randint(0,62)
	image=cv2.imread(str(testdir[t]))
	cv2.imshow(' ', image)
	cv2.waitKey(3000)
	
	image_resized= cv2.resize(image, (img_height,img_width))
	image=np.expand_dims(image_resized,axis=0)

	pred=resnet_model.predict(image)
	output_class=class_names[np.argmax(pred)]
	c += 1
	 
	print("The test result is: ", output_class)
	
	
	


#################### end of code ############################################


