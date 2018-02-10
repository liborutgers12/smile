import numpy as np
import scipy 
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers, applications
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import np_utils
from wandb.wandb_keras import WandbKerasCallback
import wandb
import smiledataset

run = wandb.init()
config = run.config

config.epochs=5
config.batch_size=16

# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

# convert classes to vector
num_classes = 2
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

train_X /= 255.0
test_X /= 255.0

print('The train data shape: ', train_X.shape)
img_rows, img_cols = train_X.shape[1:]
img_chs = 3 

# add additional dimension
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
# expand the images into three channel in order to apply the VGG19 using transfer learning
test_X = np.repeat(test_X[:, :, :, np.newaxis], img_chs, axis=3)
train_X = np.repeat(train_X[:, :, :, np.newaxis], img_chs, axis=3)
new_shape = (48,48,3)
X_test_new = np.empty(shape=(test_X.shape[0],)+new_shape)
for idx in xrange(test_X.shape[0]):
    X_test_new[idx] = scipy.misc.imresize(test_X[idx], new_shape)
X_train_new = np.empty(shape=(train_X.shape[0],)+new_shape)
for idx in xrange(train_X.shape[0]):
    X_train_new[idx] = scipy.misc.imresize(train_X[idx], new_shape)

train_X = X_train_new
test_X = X_test_new
    
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (48, 48, 3))

print('The imported VGG16 has ', len(model.layers), ' layers.')

# Freeze the layers we don't want to train. Here the first 15 layers are freezed because the smile dataset is relatively small.
for layer in model.layers:
    layer.trainable = False
    
#Adding custom Layers 
dense1 = 512
dense2 = 256

x = model.output
x = Flatten()(x)
x = Dense(dense1, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(dense2, activation="relu")(x)
x = Dropout(0.2)(x)
predicts = Dense(num_classes, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predicts)

# compile the model
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_final.summary()

model_final.fit(train_X, train_y,batch_size=config.batch_size,
    epochs=config.epochs, verbose=1,
    validation_data=(test_X, test_y), callbacks=[WandbKerasCallback()])

model_final.save("smileTransferLearning.h5")

"""
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
"""
