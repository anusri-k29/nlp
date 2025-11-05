import streamlit as st
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
import requests
import soundfile as sf
import whisper
#from audio_recorder_streamlit import audio_recorder

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Beyond Words - Emotion & Speech Analyzer",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #4A90E2; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; text-align: center; color: #7F8C8D; margin-bottom: 2rem; }
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

# Emotion model selection
if "Emotion" in analysis_mode or "Both" in analysis_mode:
    st.sidebar.subheader("Emotion Settings")
    model_choice = st.sidebar.radio(
        "Select Emotion Model:",
        ["Normal XGBoost", "Fine-tuned XGBoost"],
        index=1
    )
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)

# Speech-to-text
if "Speech-to-Text" in analysis_mode or "Both" in analysis_mode:
    st.sidebar.subheader("Speech-to-Text Settings")
    whisper_model_size = st.sidebar.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1
    )
    show_timestamps = st.sidebar.checkbox("Show Timestamps", value=False)

# ============================================
# HUGGING FACE API SETUP
# ============================================
HF_API_URLS = {
    "Normal XGBoost": "https://router.huggingface.co/hf-inference/anusrii29/xgboost-emotion",
    "Fine-tuned XGBoost": "https://router.huggingface.co/hf-inference/anusrii29/xgboost-finetuned-emotion"
}

# Securely store this in Streamlit Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if not HF_TOKEN:
    st.sidebar.warning("⚠️ Add your Hugging Face API token to .streamlit/secrets.toml as HF_TOKEN")

# ============================================
# LOAD WHISPER MODEL
# ============================================
@st.cache_resource
def load_whisper_model(model_size="base"):
    try:
        with st.spinner(f"Loading Whisper {model_size} model..."):
            model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading Whisper: {e}")
        return None

whisper_model = None
if "Speech-to-Text" in analysis_mode or "Both" in analysis_mode:
    whisper_model = load_whisper_model(whisper_model_size)

# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_handcrafted_features(y, sr, n_mfcc=40):
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

# ============================================
# HUGGING FACE API CALLER
# ============================================
def predict_via_hf_api(features, model_type="Fine-tuned XGBoost"):
    """Send handcrafted features to Hugging Face API and get predictions."""
    if not HF_TOKEN:
        st.error("Hugging Face token missing. Add it to your Streamlit secrets.")
        return None

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(HF_API_URLS[model_type], json={"features": features}, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

# ============================================
# SPEECH-TO-TEXT FUNCTION
# ============================================
def transcribe_audio(model, audio_path, with_timestamps=False):
    """Transcribe audio using Whisper, ensuring correct format"""
    try:
        # Load with librosa to ensure consistent mono 16kHz
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Re-save in standard PCM WAV format
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            sf.write(tmp_wav.name, audio, 16000, subtype='PCM_16')
            tmp_path = tmp_wav.name

        # Run Whisper transcription
        result = model.transcribe(tmp_path, word_timestamps=with_timestamps)
        os.unlink(tmp_path)
        return result

    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None
# ============================================
# PLOTS
# ============================================
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
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        st.audio(uploaded_file)

elif input_method == "🎙️ Record from Microphone":
    st.info("Record audio using the built-in Streamlit recorder below:")
    audio_bytes = st.audio_input("🎙️ Click to record your voice")

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes.getbuffer())
            audio_path = tmp_file.name
        st.audio(audio_bytes)

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

        # 📝 Speech-to-Text
        if whisper_model and ("Speech-to-Text" in analysis_mode or "Both" in analysis_mode):
            st.subheader("📝 Speech-to-Text Transcription")
            transcription_result = transcribe_audio(whisper_model, audio_path, with_timestamps=show_timestamps)
            if transcription_result:
                st.success(transcription_result['text'])

        # 🎭 Emotion Recognition
        if "Emotion" in analysis_mode or "Both" in analysis_mode:
            st.subheader("🎭 Emotion Recognition")
            feats = extract_handcrafted_features(audio, sr)
            results = predict_via_hf_api(feats, model_choice)
            if results:
                st.metric("Predicted Emotion", results['emotion'].upper())
                st.metric("Confidence", f"{results['confidence']:.2f}")

                if show_spectrogram:
                    st.pyplot(plot_spectrogram(audio, sr))

                if "probabilities" in results:
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
    "Beyond Words - XGBoost Emotion Recognition + Whisper Speech-to-Text<br>"
    "Models served via Hugging Face API"
    "</div>", unsafe_allow_html=True
)
