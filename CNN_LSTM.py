import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set image size and batch size
image_size = (48, 48)
batch_size = 32
sequence_length = 16  # Updated sequence length to match the flattened feature size

# Data augmentation and normalization for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of the data for validation
)

# Normalization for the validation data (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training images from the directory
train_generator = train_datagen.flow_from_directory(
    'FER2013/train/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'
)

# Load validation images from the directory
validation_generator = validation_datagen.flow_from_directory(
    'FER2013/test/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'
)

# Define the expression (class) you want to visualize
expression = 'surprise'

# Get the index for 'surprise'
label_map = train_generator.class_indices
surprise_label_index = label_map[expression]

# Fetch a batch of images and labels
images, labels = next(train_generator)

# Filter images for the 'angry' class
surprise_images = images[labels[:, surprise_label_index] == 1]

# Plot the first 9 'angry' images
plt.figure(figsize=(12, 12))
for i in range(min(9, surprise_images.shape[0])):  # Ensure we don't exceed available images
    plt.subplot(3, 3, i + 1)
    plt.imshow(surprise_images[i], cmap='gray')  # Use cmap='gray' for grayscale images
    plt.axis('off')  # Hide axes for better visualization
plt.suptitle(f'Sample {expression.capitalize()} Images', fontsize=16)
plt.show()


# Build the CNN + LSTM model
model = Sequential()

# First convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))  # 1 channel for grayscale
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third convolutional layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the layers before LSTM
model.add(Flatten())

# Reshape the data to be in a format suitable for LSTM
# The product of the reshaped dimensions must match the flattened output size
model.add(Reshape((16, 256)))  # Example reshaping (16 * 256 = 4096)

# LSTM layer
model.add(LSTM(128, return_sequences=False))  # Return only the final output of the LSTM
model.add(Dropout(0.5))

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer (number of classes = number of emotion categories)
model.add(Dense(train_generator.num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model with validation data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=100  # Number of epochs
)

# Save the trained model
model.save('emotion_detection_model_cnn_lstm_FER2013.h5')

# Plot accuracy and loss
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
