import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.utils import np_utils
from wandb.wandb_keras import WandbKerasCallback
import wandb
import smiledataset

run = wandb.init()
config = run.config

config.epochs=10
config.batch_size=64

# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

# convert classes to vector
num_classes = 2
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

print('The train data shape: ', train_X.shape)
img_rows, img_cols = train_X.shape[1:]

# add additional dimension
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)

train_X /= 255.0
test_X /= 255.0

dense1 = 1024
dense2 = 256

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_rows, img_cols,1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(dense1, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(dense2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

model.fit(train_X, train_y,batch_size=config.batch_size,
    epochs=config.epochs, verbose=1,
    validation_data=(test_X, test_y), callbacks=[WandbKerasCallback()])

model.save("smileCNN.h5")
