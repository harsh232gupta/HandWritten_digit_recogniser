# Handwritten Digit Recognition System Using Neural Networks

This project implements a handwritten digit recognition system using a neural network model, trained on the MNIST dataset.

## Code

```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data using mean normalization
x_train = (x_train / 255.0) - 0.5
x_test = (x_test / 255.0) - 0.5

# Flatten the data to a single-dimensional array
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Convert labels to sparse categorical format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = models.Sequential()

# Add layers to the model
model.add(layers.Dense(120, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(10, activation='linear'))  # Output layer with 10 units (one for each class)

# Compile the model with Sparse Categorical Crossentropy loss function
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the trained model
model.save('digit_recognition_model.h5')
