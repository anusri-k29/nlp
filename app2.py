import streamlit as st
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
from collections import Counter
import soundfile as sf
import whisper
import xgboost as xgb
from audio_recorder_streamlit import audio_recorder
import io

# ============================================
# PAGE CONFIG & STYLES
# ============================================
st.set_page_config(
    page_title="Beyond Words - AI Audio Analyzer",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎤 Beyond Words</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Speech Analysis: Emotion Recognition + Speech-to-Text</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR SETTINGS
# ============================================
st.sidebar.header("⚙️ Settings")

analysis_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["🎭 Emotion Recognition Only", "📝 Speech-to-Text Only", "🔥 Both (Full Analysis)"],
    index=2
)

if "Emotion" in analysis_mode or "Both" in analysis_mode:
    st.sidebar.subheader("Emotion Settings")
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)

if "Speech-to-Text" in analysis_mode or "Both" in analysis_mode:
    st.sidebar.subheader("Speech-to-Text Settings")
    whisper_model_size = st.sidebar.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    show_timestamps = st.sidebar.checkbox("Show Timestamps", value=False)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_emotion_model():
    try:
        # Load fine-tuned XGBoost model
        model = xgb.XGBClassifier()
        model.load_model('xgboost_finetuned.json')

        # Load metadata (label encoder + feature columns)
        with open('ensemble_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        le_classes = meta['label_encoder_classes']
        feature_cols = meta['feature_cols']

        # Create dummy LabelEncoder-like object
        le = type('LabelEncoder', (), {})()
        le.classes_ = np.array(le_classes)

        return model, le, feature_cols
    except Exception as e:
        st.sidebar.error(f"Error loading XGBoost model: {e}")
        return None, None, None


@st.cache_resource
def load_whisper_model(model_size="base"):
    try:
        with st.spinner(f"Loading Whisper {model_size} model... (first time may take a moment)"):
            model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading Whisper: {e}")
        return None


# Initialize models
emotion_model, le, feature_cols = None, None, None
whisper_model = None

if "Emotion" in analysis_mode or "Both" in analysis_mode:
    emotion_model, le, feature_cols = load_emotion_model()

if "Speech-to-Text" in analysis_mode or "Both" in analysis_mode:
    whisper_model = load_whisper_model(whisper_model_size)

# ============================================
# HELPER FUNCTIONS
# ============================================
def extract_handcrafted_features(y, sr, n_mfcc=40):
    """Extract handcrafted MFCC + ZCR + RMSE features"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats = {}
    for i in range(min(8, mfcc.shape[0])):
        feats[f'mfcc_mean_{i}'] = float(mfcc[i].mean())
        feats[f'mfcc_std_{i}'] = float(mfcc[i].std())
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rmse = librosa.feature.rms(y=y)
    feats['zcr_mean'] = float(np.mean(zcr))
    feats['zcr_std'] = float(np.std(zcr))
    feats['rmse_mean'] = float(np.mean(rmse))
    feats['rmse_std'] = float(np.std(rmse))
    feats['duration'] = float(len(y) / sr)
    return feats


def predict_emotion(audio, sr, model, le, feature_cols):
    feats = extract_handcrafted_features(audio, sr)
    x_input = np.array([feats.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
    probs = model.predict_proba(x_input)[0]
    pred_idx = np.argmax(probs)
    pred_emotion = le.classes_[pred_idx]
    return {
        'final_emotion': pred_emotion,
        'probabilities': dict(zip(le.classes_, probs)),
        'confidence': float(probs[pred_idx])
    }


def transcribe_audio(model, audio_path, with_timestamps=False):
    """Transcribe audio using Whisper"""
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            sf.write(tmp_wav.name, audio, sr)
            tmp_path = tmp_wav.name
        result = model.transcribe(tmp_path, word_timestamps=with_timestamps)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None


def plot_spectrogram(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_emotion_distribution(probs):
    fig, ax = plt.subplots(figsize=(10, 5))
    emotions = list(probs.keys())
    probabilities = list(probs.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    bars = ax.barh(emotions, probabilities, color=colors)
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probability Distribution')
    ax.set_xlim([0, 1])
    for bar, prob in zip(bars, probabilities):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.2%}', va='center')
    plt.tight_layout()
    return fig

# ============================================
# MAIN APP
# ============================================
st.markdown("---")
input_method = st.radio(
    "Choose Input Method:",
    [" Upload Audio File", "🎙️ Record from Microphone"],
    horizontal=True
)

audio_path = None

if input_method == " Upload Audio File":
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a']
    )
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        st.audio(uploaded_file)

elif input_method == "🎙️ Record from Microphone":
    audio_bytes = audio_recorder(
        text="Click to Record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
    )
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name
        st.audio(audio_bytes)

# ============================================
# ANALYSIS PIPELINE
# ============================================
if audio_path and st.button(" Analyze Audio"):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)

        # Speech-to-Text
        if whisper_model and ("Speech-to-Text" in analysis_mode or "Both" in analysis_mode):
            st.subheader("📝 Speech-to-Text Transcription")
            transcription_result = transcribe_audio(whisper_model, audio_path, with_timestamps=show_timestamps)
            if transcription_result:
                st.success(transcription_result['text'])

        # Emotion Recognition
        if emotion_model and ("Emotion" in analysis_mode or "Both" in analysis_mode):
            st.subheader("🎭 Emotion Recognition")
            results = predict_emotion(audio, sr, emotion_model, le, feature_cols)
            st.metric("Predicted Emotion", results['final_emotion'].upper())
            st.metric("Confidence", f"{results['confidence']:.2f}")

            if show_spectrogram:
                st.pyplot(plot_spectrogram(audio, sr))

            st.pyplot(plot_emotion_distribution(results['probabilities']))

        if os.path.exists(audio_path):
            os.unlink(audio_path)

    except Exception as e:
        st.error(f"Error: {e}")
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Beyond Words - Speech Emotion Recognition (XGBoost) + Speech-to-Text (Whisper)"
    "</div>", unsafe_allow_html=True
)
