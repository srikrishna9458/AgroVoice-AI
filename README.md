# AgroVoice AI: Rural STT Pipeline 🌾
A specialized Speech-to-Text (STT) system optimized for rural Hindi dialects, designed to bridge the communication gap for low-literacy farmers.

### 🛠️ The Technical Pipeline
1. [cite_start]**Signal Processing Layer:** Uses `librosa` and `noisereduce` (Spectral Gating) to filter machinery and wind noise[cite: 15].
2. [cite_start]**AI Inference:** Leverages `OpenAI Whisper` via a custom manual inference handler to bypass standard metadata errors[cite: 14].
3. [cite_start]**User Interface:** A real-time web interface built with `Gradio` for microphone-input testing.

### 📈 Key Results
* [cite_start]**Accuracy:** Achieved over 85% word accuracy in high-noise environments[cite: 14].
* **Latency:** Sub-second inference time on standard consumer hardware.