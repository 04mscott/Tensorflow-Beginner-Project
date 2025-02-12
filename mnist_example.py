import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

from deep_learning_models import functional_model, MyCustomModel
from my_utils import display_examples


# tensorflow.keras.Sequential
# Rarely Used
seq_model = tensorflow.keras.Sequential(
    [
        # 28x28 pixel, 1 chanel (grayscale)
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        # Takes output of conv2d and keeps the max value of n x n window
        MaxPool2D(),
        # Looks at batches and does normalization
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        # Average of values from batch normalization
        GlobalAvgPool2D(),
        # Vector containing values
        Dense(64, activation='relu'),
        # Output layer, 10 possibilities (0-9), softmax so that they are probabilities
        Dense(10, activation='softmax')
    ]
)


if __name__=='__main__':
    
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape: ", x_train.shape)
    print("y_train.shape: ", y_test.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_test.shape: ", y_test.shape)

    if False:
        display_examples(x_train, y_train)

    # Normalize Data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One Hot Encoding -> If using categorical_crossentropy loss fxn
    if False:
        y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    model = functional_model()
    # model = MyCustomModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)