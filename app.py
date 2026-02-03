import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import noisereduce as nr
import os

# 1. SETUP: Load Model & Processor
model_id = "openai/whisper-small"
print(f"Loading {model_id}...")

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. ADVICE DATABASE (Keyword Mapping for your 'AgroSarthi' logic)
ADVICE_MAP = {
    "पानी": "सिंचाई (Irrigation): अपनी मिट्टी की नमी की जांच करें। शाम को पानी देना सबसे अच्छा है।",
    "कीड़े": "कीटनाशक (Pesticide): नीम के तेल का छिड़काव करें या स्थानीय केंद्र से सलाह लें।",
    "खाद": "उर्वरक (Fertilizer): बुवाई के समय यूरिया और फास्फोरस का सही संतुलन बनाए रखें।",
    "मौसम": "मौसम (Weather): आज बारिश की संभावना है, फसल की कटाई में सावधानी बरतें।"
}

def agro_advisor(audio_path):
    if audio_path is None:
        return "No audio", "Please record your voice."

    try:
        # 3. SIGNAL PROCESSING (The Resume Layer)
        audio, rate = librosa.load(audio_path, sr=16000)
        reduced_noise = nr.reduce_noise(y=audio, sr=rate)
        
        # 4. TRANCRIPTION (Manual Inference)
        input_features = processor(reduced_noise, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # 5. ADVICE LOGIC (Keyword Search)
        advice = "क्षमा करें, मुझे इस बारे में जानकारी नहीं है। कृपया 'पानी', 'कीड़े' या 'खाद' के बारे में पूछें।"
        for key in ADVICE_MAP:
            if key in transcription:
                advice = ADVICE_MAP[key]
                break
        
        return transcription, advice

    except Exception as e:
        return "Error", str(e)

# 6. GRADIO UI DESIGN
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 AgroVoice AI: Rural Support System")
    gr.Markdown("Record your query in Hindi regarding irrigation, pests, or fertilizers.")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Here")
    
    with gr.Row():
        text_output = gr.Textbox(label="What you said (Transcription)")
        advice_output = gr.Textbox(label="AI Agricultural Advice")
    
    submit_btn = gr.Button("Get Advice")
    submit_btn.click(fn=agro_advisor, inputs=audio_input, outputs=[text_output, advice_output])

if __name__ == "__main__":
    demo.launch()