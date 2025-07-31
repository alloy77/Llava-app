# 🎙️ Multimodal RAG App – Image + Audio Interaction using OpenAI Whisper & LLaVA

A voice-interactive multimodal assistant that describes images and answers questions about them using your **voice**. It combines the power of **OpenAI Whisper** for speech recognition and **LLaVA-1.5 (Language + Vision)** for detailed visual understanding. Built with 🤗 Transformers, 🧠 Whisper, 🎨 PIL, 🔊 gTTS, and deployed via Gradio.

---

## 🚀 Features

- 🎤 **Speech-to-Text** with OpenAI Whisper (Multilingual)
- 🖼️ **Image Understanding** via LLaVA-1.5 7B (4-bit quantized for efficiency)
- 🤖 **Prompt-Driven Visual Reasoning**: Ask questions like *"What color is the microphone?"*
- 🔊 **Text-to-Speech Response** using Google TTS
- 🌐 **Gradio Web UI** for interactive multimodal querying

---

## 📦 Tech Stack

| Module            | Description                          |
|------------------|--------------------------------------|
| `transformers`    | LLaVA-1.5 image-to-text pipeline     |
| `bitsandbytes`    | 4-bit quantization for LLaVA         |
| `whisper`         | Speech-to-text (OpenAI Whisper)      |
| `gTTS`            | Text-to-speech conversion            |
| `Gradio`          | Web UI to interact with app          |
| `NLTK`            | Sentence tokenization for parsing    |
| `FFmpeg`          | Audio preprocessing support          |

---

## 🧠 How It Works

1. **User speaks** a query using a microphone.
2. **Whisper** converts the audio into text.
3. **LLaVA** receives the image and the query to generate a visual response.
4. The assistant's **reply is spoken back** using Google Text-to-Speech.
5. Everything is logged and timestamped for traceability.

---

## 📸 Sample Output

**Input Image**:  
![Sample Image](img.jpg)

**Query**:  
`"What color is the microphone in the image?"`

**LLaVA Output**:
> *"The microphone in the image is black."*

**Audio Response**:  
> 🔈 Plays generated response via gTTS.

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -q -U transformers==4.37.2
pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
pip install -q git+https://github.com/openai/whisper.git
pip install -q gradio gTTS
```

### 2. Clone and Run

```bash
git clone https://github.com/alloy77/Llava-app.git
cd Llava-app
python app.py  # or open in Jupyter/Colab
```

---

## 🧪 Run in Colab (Recommended)

- Automatically sets up everything
- Launches the Gradio app with microphone + image upload

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alloy77/Llava-app/blob/main/Multimodal_RAG.ipynb)

---

## 🧬 Components

- `img2txt()`: Combines image + prompt and extracts LLaVA output
- `transcribe()`: Converts voice input to text via Whisper
- `text_to_speech()`: Converts AI response to audio using gTTS
- `gr.Interface`: Gradio UI to glue everything together

---

## 📁 File Structure

```
Llava-app/
├── Multimodal_RAG.ipynb       # Main notebook
├── img.jpg                    # Sample image input
├── Temp.mp3 / Temp3.mp3       # Temporary audio files
├── *.txt                      # Log files
└── README.md                  # Project documentation
```

---

## ✅ TODO / Future Enhancements

- 🔐 Hugging Face Token integration for authenticated models
- 🎯 Improve prompt engineering for better reasoning
- 🗣️ Add language selection for Whisper + gTTS
- 💾 Save audio conversations for reuse
- 🧠 Connect to RAG for knowledge-grounded answers

---

## 🤝 Contributing

Open to suggestions, ideas, or collaborations! Feel free to fork and submit a PR.

