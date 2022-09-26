# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import numpy as np
import os
from PIL import Image


# sys.stdout = open("Console Output.txt","w")

Tr_directory = 'C:\Classes\SEnio temp\Training Dataset'
Test_directory = 'C:\Classes\SEnio temp\Testing Dataset'


train_DS = tf.keras.utils.image_dataset_from_directory(
    Tr_directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=2,
    image_size=(299,299),
    shuffle=True,
    seed=151,
    validation_split=0.2,
    subset="training",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

class_names = train_DS.class_names
print(class_names)


val_DS = tf.keras.utils.image_dataset_from_directory(
    Tr_directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=2,
    image_size=(299,299),
    shuffle=True,
    seed=151,
    validation_split=0.2,
    subset="validation",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299,299,3),
    pooling=None,
    classes=None,
    classifier_activation='softmax'
)
base_model.trainable = False

inputs = keras.Input(shape=(299,299,3))
inputs = tf.keras.applications.inception_v3.preprocess_input(inputs)

x = base_model(inputs, training=False)

x = keras.layers.GlobalAveragePoolingd()(x)

outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
print("model structure: ", model.summary())


model.compile(optimizer=keras.optimizer.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

history = model.fit(train_DS, validation_data=val_DS, epochs=6)
print(history.history.keys())


plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


Ptest_DS = tf.keras.utils.image_dataset_from_directory(
    Test_directory,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode='rgb',
    batch_size=2,
    image_size=(299,299),
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False,
)

test_img_dir = []
test_img = []
i = 0


DIR = "C:\Classes\SEnio temp\Testing Dataset"
for file in os.listdir("C:\Classes\SEnio temp\Testing Dataset"):
    if file.endswith(('.png', '.jpg')):
        test_img_dir.append(DIR + file)
        temp = Image.open(test_img_dir[i]).convert('RGB')
        test_img.append(temp)
        i += 1


L = test_img.__len__()
for i in range(L):
    Temp = test_img[i]
    Temp = tf.image.resize(Temp, [299,299])
    Temp = np.array(Temp, np.int32)
    plt.imshow(Temp.astype('uint8'))
    Score = (model.predict(np.expand_dims(Temp, axis=0)) > 0.5).astype("int32")
    mapper = {1: "GOOD", 0: "BAD"}
    Result = np.vectorize(mapper.get)(Score)
    plt.title(Result)
    plt.show()


    Save_DIR = "C:\Classes\SEnio temp"
    tf.keras.models.save_model(
        model,
        Save_DIR,
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None,
        options=None,
        save_traces=True
    )
# this comment for testing
