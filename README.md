# Voice Emotion Classification

## Overview
This project is a deep learning-based voice emotion classification system that detects emotions from audio recordings. It preprocesses audio files, extracts MFCC features, trains a Convolutional Recurrent Neural Network (CRNN) model, and provides a Flask API for predictions.

## Features
- **Emotion Detection**: Classifies emotions into `sad`, `neutral`, `happy`, `fear`, `angry`, `disgust`, and `surprised`.
- **Preprocessing Pipeline**: Extracts MFCC features and normalizes them.
- **Deep Learning Model**: Uses a CRNN model with Conv1D, Bidirectional LSTMs, and batch normalization.
- **Data Augmentation**: Enhances training data with techniques like Gaussian noise, pitch shifting, and time stretching.
- **Flask API**: Provides endpoints for audio file uploads and real-time microphone recording predictions.

## Project Structure
```
├── config.py          # Configuration settings
├── preprocess.py      # Preprocesses audio files (MFCC extraction, normalization)
├── model.py          # Trains and evaluates the deep learning model
├── main.py           # Flask API for emotion prediction
├── requirements.txt  # Dependencies
└── data/             # Directory for training audio files
```

## Installation
### Prerequisites
- Python 3.8+
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd voice-emotion-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Preprocess Data
Run the following command to extract and preprocess features:
```bash
python preprocess.py
```
This will save the processed data as `emotion_data.pkl`.

### 2. Train Model
Train the model using:
```bash
python model.py
```
This will save the trained model in `emotion_model.keras` and weights in `emotion_model.weights.h5`.

### 3. Run Flask API
Start the server with:
```bash
python main.py
```
The API will be available at `http://127.0.0.1:5000/`.

### 4. Make Predictions
#### Upload an Audio File
Send a POST request to `/predict` with an audio file:
```bash
curl -X POST -F "file=@path_to_audio.wav" http://127.0.0.1:5000/predict
```
#### Real-time Recording
Send a POST request to `/record`:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"duration": 10, "chunk_duration": 3}' http://127.0.0.1:5000/record
```

## Model Details
- **Feature Extraction**: Uses MFCC with padding/truncation.
- **Data Augmentation**: Adds noise, shifts time, and modifies pitch.
- **Architecture**: A combination of Conv1D, Bidirectional LSTMs, and dropout layers.
- **Loss Function**: Sparse Categorical Crossentropy.
- **Optimizer**: Adam with a step-decay learning rate.

## Dependencies
See `requirements.txt` for all dependencies. Install using:
```bash
pip install -r requirements.txt
```

## Contributing
Pull requests are welcome! Feel free to improve preprocessing, model performance, or API functionalities.

## License
This project is licensed under the MIT License.
