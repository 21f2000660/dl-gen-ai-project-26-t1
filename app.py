import os
import tempfile
import streamlit as st
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio
import torchaudio.transforms as T
from transformers import ASTForAudioClassification, AutoFeatureExtractor

# This finds the folder where app.py lives and joins it with the model path
base_path=os.path.dirname(__file__)
best_weights=os.path.join(base_path, "models", "best_ast_model.pth")
AST_VER="MIT/ast-finetuned-audioset-10-10-0.4593"
GENRES= ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
SR=16000
DURATION=10
TARGET_SAMPLES=SR*DURATION
CHUNK_DURATION = 10
HOP_DURATION = 5
CHUNK_SAMPLES = SR * CHUNK_DURATION
STEP_SAMPLES = SR * HOP_DURATION

#Caching the heavy model to avoid importing the model for every prediction
@st.cache_resource
def load_model_and_extractor():
    # Loading the same AST extractor
    ast_extractor = AutoFeatureExtractor.from_pretrained(AST_VER)

    # Initializing the AST model with modified 10-class head similar to the kaggle notebook
    model = ASTForAudioClassification.from_pretrained(
        AST_VER,
        num_labels=len(GENRES),
        ignore_mismatched_sizes=True
    )

    # map_location='cpu' to ensure it works on Streamlit servers without a GPU
    state_dict = torch.load(best_weights, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
   
    return model, ast_extractor
    
model, ast_extractor =load_model_and_extractor()

# Streamlit Application Frontend
st.title("Music Genre Classifier using AST")
st.write("Upload a .wav file to predict its genre")
uploaded_audio = st.file_uploader("Upload an audio file...", type=["wav"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format='audio/wav')
    
    with st.spinner('Predicting the genre...'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_audio.getvalue())
                tmp_path = tmp_file.name
                
            # Reading the file using torchaudio from Streamlit uploaded file buffer
            wave, sr = torchaudio.load(tmp_path, backend="soundfile")
            os.remove(tmp_path)
            
            if sr != SR:
                wave = T.Resample(sr, SR)(wave)
            
            if wave.shape[0] > 1: 
                wave = torch.mean(wave, dim=0, keepdim=True)
                
            if wave.shape[1] < TARGET_SAMPLES:
                # If audio is shorter than 10s: Center Zero-Pad
                missing_samples = TARGET_SAMPLES - wave.shape[1]
                
                # Calculate how much silence to add to the left and right
                pad_left = missing_samples // 2
                pad_right = missing_samples - pad_left
                
                # F.pad format for a 1D sequence is (pad_left, pad_right)
                wave = F.pad(wave, (pad_left, pad_right))
                
            wave = wave / (torch.max(torch.abs(wave)) + 1e-9)
            
            total_samples = wave.shape[1]
            all_chunk_logits = []

            # Step through the audio with a 5-second overlapping window
            with torch.no_grad():
                for start in range(0, total_samples, STEP_SAMPLES):
                    end = start + CHUNK_SAMPLES
                    chunk = wave[:, start:end]
                    
                    # Skipping chunk if the duration is  less than 2 seconds long
                    if chunk.shape[1] < (SR * 2) and len(all_chunk_logits) > 0:
                        break
                    
                    # Zero-pad if we hit the end of the file and the chunk is less than 10s
                    if chunk.shape[1] < CHUNK_SAMPLES:
                        chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[1]))

                    # AST Extractor expects 1D numpy array in shape - [Time, Freq] - [313, 128]
                    chunk_np = chunk.squeeze(0).numpy()
                    inputs = ast_extractor(chunk_np, sampling_rate=SR, return_tensors="pt")
                    inputs = inputs['input_values']
                    logits = model(inputs).logits
                        
                    all_chunk_logits.append(logits)

            # Average the confidence scores from all overlapping chunks
            final_logits = torch.mean(torch.stack(all_chunk_logits), dim=0)
            
            # Apply softmax to convert raw logits into percentages
            probabilities = F.softmax(final_logits, dim=1)
            
            # Get the highest score
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_genre = GENRES[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item() * 100

            st.success(f"Predicted Genre: {predicted_genre} ({confidence:.2f}% confidence)")
            
        except Exception as e:
            st.error(f"An internal error occurred: {e}")
