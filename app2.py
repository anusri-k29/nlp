import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
from collections import Counter
import soundfile as sf
import whisper
from audio_recorder_streamlit import audio_recorder
import io
# Page config
st.set_page_config(
    page_title="Beyond Words - AI Audio Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
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

# Sidebar
st.sidebar.header("⚙️ Settings")

# Mode selection
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["🎭 Emotion Recognition Only", "📝 Speech-to-Text Only", "🔥 Both (Full Analysis)"],
    index=2
)

# Emotion Recognition settings
if "Emotion" in analysis_mode or "Both" in analysis_mode:
    st.sidebar.subheader("Emotion Settings")
    segment_duration = st.sidebar.slider("Segment Duration (s)", 2.0, 5.0, 3.0, 0.5)
    overlap = st.sidebar.slider("Overlap (s)", 0.5, 2.5, 1.5, 0.5)
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)

# Speech-to-Text settings
if "Speech-to-Text" in analysis_mode or "Both" in analysis_mode:
    st.sidebar.subheader("Speech-to-Text Settings")
    whisper_model_size = st.sidebar.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    show_timestamps = st.sidebar.checkbox("Show Timestamps", value=False)

# Load Emotion Recognition Model
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('emotion_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('normalization_params.pkl', 'rb') as f:
            params = pickle.load(f)
        return model, le, params['mean'], params['std']
    except Exception as e:
        st.sidebar.error(f"Emotion model not found: {e}")
        return None, None, None, None

# Load Whisper Model
@st.cache_resource
def load_whisper_model(model_size="base"):
    try:
        with st.spinner(f"Loading Whisper {model_size} model... (first time may take a moment)"):
            model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading Whisper: {e}")
        return None

# Load models based on mode
emotion_model, le, MEAN, STD = None, None, None, None
whisper_model = None

if "Emotion" in analysis_mode or "Both" in analysis_mode:
    emotion_model, le, MEAN, STD = load_emotion_model()

if "Speech-to-Text" in analysis_mode or "Both" in analysis_mode:
    whisper_model = load_whisper_model(whisper_model_size)

# ============================================
# HELPER FUNCTIONS
# ============================================

def preprocess_audio_for_whisper(audio_array, sample_rate):
    """Preprocess audio for Whisper (16kHz mono)"""
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    
    audio_array = audio_array.astype(np.float32)
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array

def transcribe_audio(model, audio_path, with_timestamps=False):
    """Transcribe audio using Whisper (convert all to 16kHz mono WAV first)"""
    try:
        # Load audio with soundfile
        audio, sr = sf.read(audio_path)
        
        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Save temporary WAV file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            sf.write(tmp_wav.name, audio, sr)
            tmp_path = tmp_wav.name
        
        # Transcribe with Whisper
        result = model.transcribe(tmp_path, word_timestamps=with_timestamps)
        
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return result
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None


def predict_emotion(audio, sr, model, le, mean, std, segment_duration=3.0, overlap=1.5):
    """Predict emotion from audio with sliding window"""
    total_duration = len(audio) / sr
    
    if total_duration < 1.0:
        target_length = int(3.0 * sr)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    
    segment_samples = int(segment_duration * sr)
    hop_samples = int((segment_duration - overlap) * sr)
    
    all_predictions = []
    all_confidences = []
    all_probabilities = []
    
    num_segments = max(1, (len(audio) - segment_samples) // hop_samples + 1)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_segments):
        status_text.text(f"Analyzing segment {i+1}/{num_segments}...")
        start = i * hop_samples
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]
        
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
        
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        
        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]
        
        mfcc_input = (mfcc.T - mean) / (std + 1e-8)
        mfcc_input = mfcc_input[np.newaxis, ...]
        
        prediction = model.predict(mfcc_input, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx]
        
        emotion = le.classes_[predicted_idx]
        
        all_predictions.append(emotion)
        all_confidences.append(confidence)
        all_probabilities.append(prediction[0])
        
        progress_bar.progress((i + 1) / num_segments)
    
    progress_bar.empty()
    status_text.empty()
    
    vote_counts = Counter(all_predictions)
    majority_emotion = vote_counts.most_common(1)[0][0]
    
    avg_probs = np.mean(all_probabilities, axis=0)
    avg_emotion = le.classes_[np.argmax(avg_probs)]
    
    return {
        'final_emotion': avg_emotion,
        'majority_emotion': majority_emotion,
        'all_predictions': all_predictions,
        'vote_counts': vote_counts,
        'avg_probabilities': dict(zip(le.classes_, avg_probs)),
        'num_segments': num_segments,
        'duration': total_duration
    }

def plot_spectrogram(audio, sr):
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_emotion_distribution(probs):
    """Plot emotion probability distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    emotions = list(probs.keys())
    probabilities = list(probs.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    bars = ax.barh(emotions, probabilities, color=colors)
    
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probability Distribution')
    ax.set_xlim([0, 1])
    
    for bar, prob in zip(bars, probabilities):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2%}', va='center')
    
    plt.tight_layout()
    return fig

# ============================================
# MAIN APP
# ============================================

# Input method selection
st.markdown("---")
input_method = st.radio(
    "Choose Input Method:",
    [" Upload Audio File", "🎙️ Record from Microphone"],
    horizontal=True
)

audio_data = None
audio_path = None

# Handle different input methods
if input_method == " Upload Audio File":
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Upload an audio file containing speech"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(" Uploaded Audio")
            st.audio(uploaded_file, format='audio/wav')
            
            file_details = {
                "Filename": uploaded_file.name,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name

elif input_method == "🎙️ Record from Microphone":
    st.info("🎤 Click the microphone button below to start recording. Click again to stop.")
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="Click to Record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
    )
    
    if audio_bytes:
        st.success(" Recording captured!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎙️ Recorded Audio")
            st.audio(audio_bytes, format='audio/wav')
        
        # Save recorded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name

# Analyze button
if audio_path and st.button(" Analyze Audio", type="primary", use_container_width=True):
    
    with st.spinner(" Processing audio..."):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            st.markdown("---")
            
            # ===== SPEECH-TO-TEXT =====
            if whisper_model and ("Speech-to-Text" in analysis_mode or "Both" in analysis_mode):
                st.subheader(" Speech-to-Text Transcription")
                
                with st.spinner("Transcribing..."):
                    transcription_result = transcribe_audio(whisper_model, audio_path, show_timestamps)
                
                if transcription_result:
                    # Display transcription
                    st.markdown("###  Transcribed Text:")
                    st.success(transcription_result['text'])
                    
                    # Show timestamps if enabled
                    if show_timestamps and 'segments' in transcription_result:
                        with st.expander("⏱️ View Timestamps"):
                            for segment in transcription_result['segments']:
                                start = segment['start']
                                end = segment['end']
                                text = segment['text']
                                st.write(f"**[{start:.2f}s - {end:.2f}s]:** {text}")
                    
                    # Detected language
                    if 'language' in transcription_result:
                        st.info(f" Detected Language: **{transcription_result['language'].upper()}**")
                
                st.markdown("---")
            
            # ===== EMOTION RECOGNITION =====
            if emotion_model and ("Emotion" in analysis_mode or "Both" in analysis_mode):
                st.subheader(" Emotion Recognition")
                
                emotion_results = predict_emotion(
                    audio, sr, emotion_model, le, MEAN, STD,
                    segment_duration=segment_duration,
                    overlap=overlap
                )
                
                # Display emotion results
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric(" Predicted Emotion", emotion_results['final_emotion'].upper())
                
                with col_res2:
                    st.metric(" Duration", f"{emotion_results['duration']:.2f}s")
                
                with col_res3:
                    st.metric(" Segments", emotion_results['num_segments'])
                
                st.markdown("---")
                
                # Detailed emotion analysis
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.markdown("**Segment Distribution:**")
                    for emotion, count in emotion_results['vote_counts'].most_common():
                        percentage = (count / emotion_results['num_segments']) * 100
                        st.write(f"• **{emotion.capitalize()}**: {count}/{emotion_results['num_segments']} ({percentage:.1f}%)")
                
                with col_detail2:
                    st.markdown("**Confidence Scores:**")
                    sorted_probs = sorted(emotion_results['avg_probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    for emotion, prob in sorted_probs:
                        st.write(f"• **{emotion.capitalize()}**: {prob:.2%}")
                
                # Visualizations
                st.markdown("---")
                st.subheader(" Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    fig_dist = plot_emotion_distribution(emotion_results['avg_probabilities'])
                    st.pyplot(fig_dist)
                
                with viz_col2:
                    if show_spectrogram:
                        fig_spec = plot_spectrogram(audio, sr)
                        st.pyplot(fig_spec)
            
            # Cleanup
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        
        except Exception as e:
            st.error(f" Error during analysis: {e}")
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

# Instructions
else:
    if not audio_path:
        st.info(" Upload an audio file or record from microphone to get started!")
        
        st.markdown("---")
        st.subheader(" About")
        
        col_about1, col_about2 = st.columns(2)
        
        with col_about1:
            st.markdown("""
            ### 🎭 Emotion Recognition
            Analyzes speech to detect emotions:
            - 😐 Neutral
            - 😌 Calm
            - 😊 Happy
            - 😢 Sad
            - 😠 Angry
            - 😨 Fearful
            - 🤢 Disgust
            - 😲 Surprised
            """)
        
        with col_about2:
            st.markdown("""
            ### 📝 Speech-to-Text
            Converts speech to text using OpenAI Whisper:
            - Multi-language support
            - High accuracy transcription
            - Optional timestamps
            - Works with various accents
            """)
        
        st.markdown("---")
        st.markdown("""
        ###  Tips
        - Use clear audio for best results
        - Microphone recording works best in quiet environments
        - Longer audio files are analyzed in segments
        - Adjust settings in the sidebar for fine-tuning
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Beyond Words -  Speech Emotion Recognition + Speech-to-Text"
    "</div>",
    unsafe_allow_html=True
)
