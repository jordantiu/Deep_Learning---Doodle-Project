import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils

#Using google dataset of numpy bitmaps

#testing with single classifier

cat = np.load('full_numpy_bitmap_cat.npy')
sun = np.load('full_numpy_bitmap_sun.npy')
fish = np.load('full_numpy_bitmap_fish.npy')

#Make into readable dataframe or else accuracy will stay 1
cat = np.c_[cat, np.zeros(len(cat))]
sun = np.c_[sun, np.ones(len(sun))]
fish = np.c_[fish, 2*np.ones(len(fish))]


#merging the two different numpys for training and test data
X = np.concatenate((cat[:50000,:-1], sun[:50000,:-1], fish[:50000,:-1]), axis=0).astype('float32')/255 #scale data
y = np.concatenate((cat[:50000, -1], sun[:50000,-1], fish[:50000, -1]), axis= 0).astype('float32')

y= np_utils.to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


num_classes = 3

#google data is provided in 28x28 so reshape data to enter into model
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#make image
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
write_graph=True, write_images=True)

#creating model
model = Sequential()
#Using 3x3 because images are very simple, no need for a large filter
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Flatten())
model.add(Dense(518, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=100, callbacks=[tbCallBack])


scores = model.evaluate(X_test, y_test, verbose=0)
model.save("model.h5")

print("Accuracy: %.2f%%" % (scores[1]*100))

