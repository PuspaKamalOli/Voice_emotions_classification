class Config:
    EMOTIONS = ['sad', 'neutral', 'happy', 'fear', 'angry', 'disgust','surprised']
    DRIVE_PATH = './data'
    MAX_PAD_LEN = 300
    SAMPLE_RATE = 16000
    MODEL_PATH = 'emotion_model.keras'
    WEIGHTS_PATH = 'emotion_model.weights.h5'
    DATA_PATH = 'emotion_data.pkl'
    UPLOAD_FOLDER = 'uploads'