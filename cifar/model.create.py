# -*- coding: utf-8 -*
import keras
import matplotlib.pylab as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import cifar10
NUM_CLASSES=10
BATCH_SIZE=32
EPOCHS=int(input("Epochs as Integer: "))
CIFAR_10_CLASSES=["Plane","Car","bird","cat","deer","dog","frog","horse","ship","truck"]
(images_train,labels_train),(images_test,labels_test)=cifar10.load_data()
trys=49
plt.title("CALL FOR TEST: "+CIFAR_10_CLASSES[int(labels_train[trys])])
plt.imshow(images_train[trys])
plt.show()
images_train=np.array(images_train,dtype="float32")
images_test=np.array(images_test,dtype="float32")
images_train/=255
images_test/=255
labels_train=to_categorical(labels_train, NUM_CLASSES)
labels_test=to_categorical(labels_test, NUM_CLASSES)
#KI MODEL
inputs = Input(shape=(32, 32, 3))
drop_out = 0.2
x = Convolution2D(32, 3, activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(drop_out)(x)
x = Convolution2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(drop_out)(x)
x = Convolution2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(drop_out)(x)
x = Convolution2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Convolution2D(64, 3, activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(input=inputs, output=x)
summary=model.summary()
print(summary)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images_train,labels_train,batch_size=BATCH_SIZE,epochs=EPOCHS)
scores=model.evaluate(images_train,labels_train)
print("Loss:"+str(scores[0]*100))
print("Accuracy: "+str(scores[1]*100))
print("Save Model:")
model.save("cifar-model.h5")
