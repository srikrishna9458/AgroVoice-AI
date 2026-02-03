import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import noisereduce as nr
import soundfile as sf
import os
import io

# 1. SETUP
file_path = "test_audio.wav" 
model_id = "openai/whisper-small" 

def run_transcription():
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    # 2. LOAD PROCESSOR AND MODEL (Manual approach)
    print(f"Loading Whisper model: {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 3. PRE-PROCESSING
    print(f"Loading and cleaning: {file_path}...")
    audio, rate = librosa.load(file_path, sr=16000)
    
    print("Applying noise reduction (Spectral Gating)...")
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=rate)

    # 4. MANUAL INFERENCE (Avoids the KeyError)
    print("Transcribing...")
    
    # Convert cleaned audio to features
    input_features = processor(reduced_noise_audio, sampling_rate=16000, return_tensors="pt").input_features 
    input_features = input_features.to(device)

    # Generate token ids
    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n" + "="*30)
    print("TRANSCRIPTION RESULT")
    print("="*30)
    print(transcription)
    print("="*30 + "\n")

if __name__ == "__main__":
    run_transcription()