import os
import pickle
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from config import Config

class DataPreprocessor:
    """
    A class used to preprocess audio data for emotion detection.

    Attributes:
    ----------
    emotions : list
        A list of emotions to be detected.
    drive_path : str
        The path to the directory containing the audio files.
    max_pad_len : int
        The maximum length to which MFCC features will be padded or truncated.

    Methods:
    -------
    prepare_data()
        Prepares the data by extracting MFCC features and normalizing them.
    process_file(file_path)
        Processes a single audio file and extracts MFCC features.
    _extract_mfcc(audio, sr)
        Extracts MFCC features from an audio array.
    _normalize(X)
        Normalizes the MFCC features.
    _shuffle_data(X, y)
        Shuffles the data.
    """

    def __init__(self, emotions, drive_path, max_pad_len):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Parameters:
        ----------
        emotions : list
            A list of emotions to be detected.
        drive_path : str
            The path to the directory containing the audio files.
        max_pad_len : int
            The maximum length to which MFCC features will be padded or truncated.
        """
        self.emotions = emotions
        self.drive_path = drive_path
        self.max_pad_len = max_pad_len

    def prepare_data(self):
        """
        Prepares the data by extracting MFCC features from audio files, normalizing them,
        and shuffling the dataset.

        Returns:
        -------
        tuple
            A tuple containing the features (X) and labels (y).
        """
        X, y = [], []
        for emotion in self.emotions:
            emotion_path = os.path.join(self.drive_path, emotion)
            for filename in os.listdir(emotion_path):
                if filename.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(emotion_path, filename)
                    try:
                        mfcc = self.process_file(file_path)
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(self.emotions.index(emotion))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        X, y = np.array(X), np.array(y)
        X = self._normalize(X)
        return self._shuffle_data(X, y)

    def process_file(self, file_path):
        """
        Processes a single audio file by loading it and extracting MFCC features.

        Parameters:
        ----------
        file_path : str
            The path to the audio file to be processed.

        Returns:
        -------
        np.ndarray
            The extracted MFCC features.
        """
        audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        return self._extract_mfcc(audio, sr)

    def _extract_mfcc(self, audio, sr):
        """
        Extracts MFCC features from an audio array.

        Parameters:
        ----------
        audio : np.ndarray
            The audio array.
        sr : int
            The sample rate of the audio.

        Returns:
        -------
        np.ndarray
            The MFCC features with shape (n_mfcc, max_pad_len).
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = self.max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_pad_len]
        return mfcc

    def _normalize(self, X):
        """
        Normalizes the MFCC features.

        Parameters:
        ----------
        X : np.ndarray
            The MFCC features to be normalized.

        Returns:
        -------
        np.ndarray
            The normalized MFCC features.
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return X

    def _shuffle_data(self, X, y):
        """
        Shuffles the data.

        Parameters:
        ----------
        X : np.ndarray
            The features.
        y : np.ndarray
            The labels.

        Returns:
        -------
        tuple
            A tuple containing the shuffled features (X) and labels (y).
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

def save_data(X, y, file_path):
    """
    Saves the data to a file.

    Parameters:
    ----------
    X : np.ndarray
        The features.
    y : np.ndarray
        The labels.
    file_path : str
        The path to the file where the data will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump((X, y), f)

if __name__ == "__main__":
    preprocessor = DataPreprocessor(Config.EMOTIONS, Config.DRIVE_PATH, Config.MAX_PAD_LEN)
    X, y = preprocessor.prepare_data()
    save_data(X, y, Config.DATA_PATH)
    print("Data preparation complete and saved to emotion_data.pkl")