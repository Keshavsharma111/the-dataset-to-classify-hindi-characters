# the-dataset-to-classify-hindi-characters
Devanagari Hindi MNIST Images Dataset
# Devanagari Handwritten Character Recognition

This project focuses on recognizing handwritten Devanagari characters using a Convolutional Neural Network (CNN). The dataset includes 92,000 images across 46 classes from the Devanagari script.
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
train_path = '/content/Hindi/Train'
test_path = '/content/Hindi/Test'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(32, 32),
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(32, 32),
    color_mode='grayscale',
    class_mode='categorical'
)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(46, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
sample_images, sample_labels = next(test_generator)
predictions = model.predict(sample_images)

for i in range(5):
    plt.imshow(sample_images[i].reshape(32, 32), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(sample_labels[i])}")
    plt.axis('off')
    plt.show()
model.save("devanagari_character_model.h5")
