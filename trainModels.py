from google.colab import drive
import tensorflow as tf
from numpy import genfromtxt

# Mount Google Drive
drive.mount('/content/drive')

# Update the paths to match your Google Drive directory structure
trainingX_path = '/content/drive/My Drive/enter your path/BackGammonAI/datasets/trainingX.csv'
trainingY_path = '/content/drive/My Drive/enter your path/BackGammonAI/datasets/trainingY.csv'
testX_path = '/content/drive/My Drive/enter your path/BackGammonAI/datasets/testX.csv'
testY_path = '/content/drive/My Drive/enter your path/BackGammonAI/datasets/testY.csv'

# Load datasets
trainingX = genfromtxt(trainingX_path, delimiter=',')
trainingY = genfromtxt(trainingY_path)
trainingX = tf.keras.utils.normalize(trainingX, axis=1)

testX = genfromtxt(testX_path, delimiter=',')
testY = genfromtxt(testY_path)
testX = tf.keras.utils.normalize(testX, axis=1)

# Function to create and compile a model
def create_model(layers, neurons, activation='relu', output_activation='softmax'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons, input_dim=31, activation=activation))
    for _ in range(layers - 1):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(601, activation=output_activation))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
    return model

# Easy Model: 2 layers, 64 neurons each
easy_model = create_model(layers=2, neurons=64)
easy_model_save_path = '/content/drive/My Drive/enter your path/models/easy_model.keras'
easy_model.fit(trainingX, trainingY, epochs=10, batch_size=32, validation_split=0.15)
easy_model.evaluate(testX, testY)
easy_model.save(easy_model_save_path)

# Medium Model: 3 layers, 128 neurons each
medium_model = create_model(layers=3, neurons=128)
medium_model_save_path = '/content/drive/My Drive/enter your path/models/medium_model.keras'
medium_model.fit(trainingX, trainingY, epochs=10, batch_size=32, validation_split=0.15)
medium_model.evaluate(testX, testY)
medium_model.save(medium_model_save_path)

# Hard Model: 4 layers, 256 neurons each
hard_model = create_model(layers=4, neurons=256)
hard_model_save_path = '/content/drive/My Drive/enter your path/models/hard_model.keras'
hard_model.fit(trainingX, trainingY, epochs=10, batch_size=32, validation_split=0.15)
hard_model.evaluate(testX, testY)
hard_model.save(hard_model_save_path)
