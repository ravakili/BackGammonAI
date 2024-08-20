# BackGammon-AI-Model

This repository contains the code and models for a Backgammon AI that provides move predictions at three different difficulty levels: easy, medium, and hard. The AI models are built using TensorFlow and trained on Backgammon game data. The repository includes scripts for training the models and for generating predictions from a given board state.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Generating Predictions](#generating-predictions)
- [Model Details](#model-details)
- [Contributing](#contributing)

## Overview

The Backgammon AI project allows users to classify Backgammon board states and get the top move predictions based on the AI's understanding of the game. Three models of varying complexity (easy, medium, and hard) are included to cater to different levels of gameplay sophistication.

## Project Structure
```bash
├── datasets/
│ ├── trainingX.csv
│ ├── trainingY.csv
│ ├── testX.csv
│ └── testY.csv
├── labels/
│ └── labels.txt
├── models/
│ ├── easy_model.keras
│ ├── medium_model.keras
│ └── hard_model.keras
├── trainModels.py
├── getTop10Moves.py
└── README.md
```

- `datasets/`: Contains the training and test datasets.
- `labels/`: Includes the labels file that maps model outputs to moves.
- `models/`: Contains the pre-trained models for easy, medium, and hard difficulties.
- `trainModels.py`: Script for training the models.
- `getTop10Moves.py`: Script for generating predictions from a given board state.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Models
To train the models, run the `train_models.py` script:

```bash
python train_models.py
```
This will train the easy, medium, and hard models using the data in the datasets/ directory. The trained models will be saved in the models/ directory.

Generating Predictions
To generate predictions for a given board state, run the predict_moves.py script:

```bash
python predict_moves.py
```
You'll be prompted to input the board state as a comma-separated string of values. The script will output the top 10 move predictions from all three difficulty models (easy, medium, and hard).

## Model Details
### Easy Model:
Architecture: 2 layers, 64 neurons each
Purpose: Quick and less accurate predictions
### Medium Model:
Architecture: 3 layers, 128 neurons each
Purpose: Balanced performance with reasonable accuracy
### Hard Model:
Architecture: 4 layers, 256 neurons each
Purpose: Complex and accurate predictions, suitable for advanced gameplay
## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are encouraged.
