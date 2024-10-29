import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


# Set image size and batch size
image_size = (48, 48)  # Assuming images are 48x48, adjust if necessary
batch_size = 32

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Normalization for validation (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training images from the directory
train_generator = train_datagen.flow_from_directory(
    'FER2013/train/',  # Path to your dataset
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'
)

# Load validation images from the directory
validation_generator = validation_datagen.flow_from_directory(
    'FER2013/test/',  # Path to your dataset
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'
)


# Define the expression (class) you want to visualize
expression = 'angry'

# Get the index for 'angry'
label_map = train_generator.class_indices
angry_label_index = label_map[expression]

# Fetch a batch of images and labels
images, labels = next(train_generator)

# Filter images for the 'angry' class
angry_images = images[labels[:, angry_label_index] == 1]

# Plot the first 9 'angry' images
plt.figure(figsize=(12, 12))
for i in range(min(9, angry_images.shape[0])):  # Ensure we don't exceed available images
    plt.subplot(3, 3, i + 1)
    plt.imshow(angry_images[i], cmap='gray')  # Use cmap='gray' for grayscale images
    plt.axis('off')  # Hide axes for better visualization
plt.suptitle(f'Sample {expression.capitalize()} Images', fontsize=16)
plt.show()

# Define the dual-pathway CNN model structure
input_layer = Input(shape=(48, 48, 1))

# Pathway 1
pathway1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
pathway1 = BatchNormalization()(pathway1)
pathway1 = MaxPooling2D(pool_size=(2, 2))(pathway1)
pathway1 = Dropout(0.25)(pathway1)

pathway1 = Conv2D(128, (3, 3), activation='relu')(pathway1)
pathway1 = BatchNormalization()(pathway1)
pathway1 = MaxPooling2D(pool_size=(2, 2))(pathway1)
pathway1 = Dropout(0.25)(pathway1)

pathway1 = Flatten()(pathway1)

# Pathway 2
pathway2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
pathway2 = BatchNormalization()(pathway2)
pathway2 = MaxPooling2D(pool_size=(2, 2))(pathway2)
pathway2 = Dropout(0.25)(pathway2)

pathway2 = Conv2D(128, (3, 3), activation='relu')(pathway2)
pathway2 = BatchNormalization()(pathway2)
pathway2 = MaxPooling2D(pool_size=(2, 2))(pathway2)
pathway2 = Dropout(0.25)(pathway2)

pathway2 = Flatten()(pathway2)

# Concatenate both pathways
concatenated = Concatenate()([pathway1, pathway2])

# Fully connected layer after concatenation
dense_layer = Dense(128, activation='relu')(concatenated)
dense_layer = Dropout(0.5)(dense_layer)

# Output layer (number of classes = number of emotion categories)
output_layer = Dense(train_generator.num_classes, activation='softmax')(dense_layer)

# Build and compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model with validation data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=100
)

# Save the trained model
model.save('emotion_detection_model_dualCNN_pathway.h5')

# Plot accuracy and loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
