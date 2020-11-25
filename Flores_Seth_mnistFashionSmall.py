import tensorflow as tf
import numpy as np
import os
import distutils

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension and scale to [0,1]
x_train = np.expand_dims(x_train, -1)/255
x_test = np.expand_dims(x_test, -1)/255

n_train = 1000
np.random.seed(2020)
ind = np.random.permutation(x_train.shape[0])
x_unlabeled =  x_train[ind[n_train:]]

x_train =  x_train[ind[:n_train]]
y_train = y_train[ind[:n_train]]

# Convert y to one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
#_______________________________________________________________________

xs = np.zeros_like(x_train)
xs[:,:-1,:] = x_train[:,1:,:]#shift all images up 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs1 = np.zeros_like(x_train)
xs1[:,1:,:] = x_train[:,:-1,:]#shift all images down 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs2 = np.zeros_like(x_train)
xs2[:,:,1:] = x_train[:,:,:-1]#shift all images right 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs3 = np.zeros_like(x_train)
xs3[:,:,:-1] = x_train[:,:,1:]#shift all images left 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs4 = np.zeros_like(x_train)
xs4[:,:-1,:-1] = x_train[:,1:,1:]#shift all images up left 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs5 = np.zeros_like(x_train)
xs5[:,:-1,1:] = x_train[:,1:,:-1]#shift all images up right 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs6 = np.zeros_like(x_train)
xs6[:,1:,:-1] = x_train[:,:-1,1:]#shift all images down left 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs7 = np.zeros_like(x_train)
xs7[:,1:,1:] = x_train[:,:-1,:-1]#shift all images down right 1
# plt.imshow(xs[0].reshape(28,28),"gray")
xs_train = np.zeros_like(x_train)
#now average the images and use that to train
xs_train = (x_train[:,:,:]+xs[:,:,:]+xs1[:,:,:]+xs2[:,:,:]+xs3[:,:,:]+xs4[:,:,:]+xs5[:,:,:]+xs6[:,:,:]+xs7[:,:,:])/9
xn_train = np.zeros_like(x_train,shape=(2*x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_train=xs_train

#______________________________________________________________________
import tensorflow as tf
from tensorflow.keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
import os
import distutils

def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64,activation ='relu'))
  model.add(tf.keras.layers.Dense(10,activation = 'softmax'))
  return model
#_______________________________________________________________________
model = create_model()
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=300, epochs=20, validation_data=(x_test, y_test), shuffle= False)
#________________________________________________________________________
y_unlabeled = model.predict(x_unlabeled)#predict unlabeled data
x_train = np.vstack((x_train,x_unlabeled))#combine data
y_train = np.vstack((y_train,y_unlabeled))
n_train = 1000
np.random.seed(2020)
ind = np.random.permutation(x_train.shape[0])
x_train =  x_train[ind[:n_train]]#now take 1000 random portions of the combined data to retrain with
y_train = y_train[ind[:n_train]]
#_________________________________________________________________________
#remake the model with the new unlabeled data set
model = create_model()
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=500, epochs=20, validation_data=(x_test,y_test), shuffle= False)