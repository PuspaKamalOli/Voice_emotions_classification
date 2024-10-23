# ##for testig
# # curl -X POST -F "file=@//Users/puspakamaloli/Desktop/TESS_CREMA/data/neutral/1001_MTI_NEU_XX.wav" http://127.0.0.1:5000/predict
from flask import Flask, request, jsonify, Response
import numpy as np
import os
import librosa
from tensorflow.keras.models import load_model
import sounddevice as sd
import time
from preprocess import Config, DataPreprocessor

app = Flask(__name__)

# Load the trained model
model = load_model(Config.MODEL_PATH)

def extract_mfcc(audio, sr=Config.SAMPLE_RATE, max_pad_len=Config.MAX_PAD_LEN):
    """
    Extract MFCC features from an audio array.

    Parameters:
        audio (np.ndarray): Audio array.
        sr (int): Sample rate.
        max_pad_len (int): Maximum length to pad/truncate the MFCC features.

    Returns:
        np.ndarray: MFCC features with shape (n_mfcc, max_pad_len).
    """
    audio = audio.astype(np.float32)  # Convert to floating-point
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc

def predict_emotion_from_audio(audio):
    """
    Predict emotion from an audio array.

    Parameters:
        audio (np.ndarray): Audio array.

    Returns:
        int: Index of the predicted emotion.
    """
    mfcc = extract_mfcc(audio)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    prediction = model.predict(mfcc)
    predicted_class_index = np.argmax(prediction)

    return predicted_class_index

def record_and_predict(duration=5, chunk_duration=3, fs=Config.SAMPLE_RATE):
    """
    Record audio in chunks and predict emotion for each chunk.

    Parameters:
        duration (int): Total duration to record in seconds.
        chunk_duration (int): Duration of each chunk in seconds.
        fs (int): Sampling rate.

    Returns:
        list: Predicted emotions for each chunk.
    """
    num_chunks = duration // chunk_duration
    predictions = []
    
    for _ in range(num_chunks):
        audio = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        audio = audio.flatten()  # Flatten the array
        predicted_emotion_index = predict_emotion_from_audio(audio)
        predictions.append(Config.EMOTIONS[predicted_emotion_index])
        time.sleep(chunk_duration)
    
    return predictions

# Define the emotion labels
emotions = Config.EMOTIONS

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from an uploaded audio file.

    Returns:
        JSON: Predicted emotion.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)

        audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        predicted_emotion_index = predict_emotion_from_audio(audio)
        predicted_emotion = emotions[predicted_emotion_index]
        
        os.remove(file_path)  # Clean up the file after prediction
        
        return jsonify({"emotion": predicted_emotion})

@app.route('/record', methods=['POST'])
def record():
    """
    Record audio from the microphone and predict emotion in real-time chunks.

    Returns:
        JSON: Predicted emotions for each chunk.
    """
    duration = request.json.get('duration', 15)  # Default duration is 15 seconds
    chunk_duration = request.json.get('chunk_duration', 3)  # Default chunk duration is 3 seconds
    
    predictions = record_and_predict(duration=duration, chunk_duration=chunk_duration)
    
    return jsonify({"emotions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
