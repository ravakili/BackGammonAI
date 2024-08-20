from google.colab import drive
import tensorflow as tf
import numpy as np

# Mount Google Drive
drive.mount('/content/drive')

# Update the paths to match your Google Drive directory structure
labels_path = '/content/drive/My Drive/enter your path/BackGammonAI/labels/labels.txt'
easy_model_path = '/content/drive/My Drive/enter your path/BackGammonAI/models/easy_model.keras'
medium_model_path = '/content/drive/My Drive/enter your path/BackGammonAI/models/medium_model.keras'
hard_model_path = '/content/drive/My Drive/enter your path/BackGammonAI/models/hard_model.keras'

count = 0
outputMapping = {}

# Open the labels file
with open(labels_path) as file:
    for line in file:
        outputMapping[count] = line.strip('\n')
        count += 1

# Load all models
easy_model = tf.keras.models.load_model(easy_model_path)
medium_model = tf.keras.models.load_model(medium_model_path)
hard_model = tf.keras.models.load_model(hard_model_path)

# Get board input from user
print("Please provide a board to classify moves for (comma-separated values):")
boardString = input()

singularInputs = boardString.split(",")
input_array = np.zeros(31)

for i in range(0, 31):
    input_array[i] = float(singularInputs[i])

input_array = tf.keras.utils.normalize(input_array.reshape(1, -1))  # Reshape to match input dimensions

# Generate predictions from all three models
def get_top_predictions(model, input_array, outputMapping, difficulty):
    predictions = model.predict(input_array)
    topTenVals = np.sort(predictions[0])[-10:]
    topTen = np.argsort(predictions[0])[-10:]
    print(f"------------------------\nTop 10 Predictions for {difficulty.capitalize()} Model\n------------------------")
    for i in range(9, -1, -1):
        print(f"{abs(i-10)}: {outputMapping[topTen[i]]} : {topTenVals[i]}")
        print("------------------------")

# Get and display predictions for each difficulty level
get_top_predictions(easy_model, input_array, outputMapping, 'easy')
get_top_predictions(medium_model, input_array, outputMapping, 'medium')
get_top_predictions(hard_model, input_array, outputMapping, 'hard')
