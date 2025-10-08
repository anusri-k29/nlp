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

# Page config
st.set_page_config(
    page_title="Beyond Words - Speech Emotion Recognition",
    layout="wide"
)

# Title and description
st.title("Beyond Words - Speech Emotion Recognition")
st.markdown("Upload an audio file to detect the emotion in speech!")

# Load model and encoder (you need to save these first!)
@st.cache_resource
def load_model_and_encoder():
    try:
        model = load_model('emotion_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('normalization_params.pkl', 'rb') as f:
            params = pickle.load(f)
        return model, le, params['mean'], params['std']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure 'emotion_model.h5', 'label_encoder.pkl', and 'normalization_params.pkl' are in the same directory.")
        return None, None, None, None

model, le, MEAN, STD = load_model_and_encoder()

def extract_features_fixed_length(audio, sr, max_len=130):
    """Extract MFCC features with fixed temporal dimension"""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        
        return mfcc.T
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def predict_emotion(audio, sr, model, le, mean, std, segment_duration=3.0, overlap=1.5):
    """Predict emotion from audio with sliding window"""
    total_duration = len(audio) / sr
    
    # Handle very short audio
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
    
    for i in range(num_segments):
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
    
    # Aggregate results
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

# Main app
if model is not None and le is not None:
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3'],
        help="Upload an audio file containing speech"
    )
    
    if uploaded_file is not None:
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Audio File")
            st.audio(uploaded_file, format='audio/wav')
            
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
        
        # Process button
        if st.button(" Analyze Emotion", type="primary", use_container_width=True):
            
            with st.spinner("Processing audio... This may take a moment."):
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load audio
                    audio, sr = librosa.load(tmp_path, sr=22050)
                    
                    # Predict emotion
                    results = predict_emotion(
                        audio, sr, model, le, MEAN, STD,
                        segment_duration=segment_duration,
                        overlap=overlap
                    )
                    
                    # Display results
                    st.success(" Analysis Complete!")
                    
                    # Main result
                    st.markdown("---")
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.metric(" Predicted Emotion", results['final_emotion'].upper())
                    
                    with col_res2:
                        st.metric("⏱️ Duration", f"{results['duration']:.2f}s")
                    
                    with col_res3:
                        st.metric("📊 Segments Analyzed", results['num_segments'])
                    
                    st.markdown("---")
                    
                    # Detailed analysis
                    if show_details:
                        st.subheader("Detailed Analysis")
                        
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.markdown("**Segment-wise Distribution:**")
                            for emotion, count in results['vote_counts'].most_common():
                                percentage = (count / results['num_segments']) * 100
                                st.write(f"• **{emotion.capitalize()}**: {count}/{results['num_segments']} ({percentage:.1f}%)")
                        
                        with col_detail2:
                            st.markdown("**Confidence Scores:**")
                            sorted_probs = sorted(results['avg_probabilities'].items(), 
                                                key=lambda x: x[1], reverse=True)
                            for emotion, prob in sorted_probs:
                                st.write(f"• **{emotion.capitalize()}**: {prob:.2%}")
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("Visualizations")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Emotion distribution
                        fig_dist = plot_emotion_distribution(results['avg_probabilities'])
                        st.pyplot(fig_dist)
                    
                    with viz_col2:
                        if show_spectrogram:
                            # Spectrogram
                            fig_spec = plot_spectrogram(audio, sr)
                            st.pyplot(fig_spec)
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    else:
        # Instructions
        st.info("Upload an audio file to get started!")
        
        st.markdown("---")
        st.subheader("Instructions")
        st.markdown("""
        1. **Upload** an audio file (WAV or MP3)
        2. **Adjust** settings in the sidebar if needed
        3. **Click** the "Analyze Emotion" button
        4. **View** the predicted emotion and detailed analysis
        
        ### Supported Emotions:
        - 😐 Neutral
        - 😌 Calm
        - 😊 Happy
        - 😢 Sad
        - 😠 Angry
        - 😨 Fearful
        - 🤢 Disgust
        - 😲 Surprised
        """)
        
        st.markdown("---")
        st.subheader("Tips")
        st.markdown("""
        - For best results, use clear speech audio
        - Longer audio files are analyzed in segments
        - The app works with various audio formats
        """)

else:
    st.error(" Model not loaded. Please check if model files exist.")
    st.info("""
    **Required files:**
    - `emotion_model.h5` (trained model)
    - `label_encoder.pkl` (label encoder)
    - `normalization_params.pkl` (mean and std values)
    
    **To create these files, add this code after training:**
    ```python
    # Save model
    model.save('emotion_model.h5')
    
    # Save label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save normalization parameters
    with open('normalization_params.pkl', 'wb') as f:
        pickle.dump({'mean': MEAN, 'std': STD}, f)
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built using Streamlit | NLP Project "
    "</div>",
    unsafe_allow_html=True
)
