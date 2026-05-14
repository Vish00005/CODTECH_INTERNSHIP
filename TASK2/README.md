# 🎙️ AI Speech Recognition System

> A Python-based speech-to-text tool that transcribes short audio clips using two industry-leading pre-trained models: **Google Web Speech API** and **OpenAI Whisper**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [FFmpeg Setup](#ffmpeg-setup)
- [Usage](#usage)
- [Supported Audio Formats](#supported-audio-formats)
- [Example Output](#example-output)
- [Error Handling](#error-handling)
- [Optional Enhancements](#optional-enhancements)

---

## 📖 Project Overview

The **AI Speech Recognition System** converts audio files into text using two transcription backends:

| Method | Internet Required | Speed | Accuracy |
|---|---|---|---|
| Google SpeechRecognition | ✅ Yes | ⚡ Fast | Good |
| OpenAI Whisper (offline) | ❌ No | 🐢 Moderate | Excellent |

Whisper automatically downloads the selected model on first use and caches it locally for subsequent runs.

---

## ✨ Features

- 🔊 **Dual transcription methods** – Google Web Speech API & OpenAI Whisper
- 🎵 **Multi-format support** – `.wav`, `.mp3`, `.m4a`
- ⏱️ **Audio duration & processing time** displayed for every run
- 🌐 **Language detection** – Whisper automatically identifies the spoken language
- 📍 **Timestamp output** – Optional per-segment timestamps from Whisper
- 💾 **Auto-save** – Transcription saved to `output/transcription_output.txt` with metadata
- 🛡️ **Robust error handling** – Friendly messages for missing files, no internet, FFmpeg not found, etc.
- 🧩 **Modular design** – Clean, well-commented functions suitable for beginners

---

## 📁 Project Structure

```
TASK2/
└── speech_recognition_system/
    ├── speech_to_text.py          # Main script – all functions + CLI
    ├── requirements.txt           # Python dependencies
    ├── README.md                  # This file
    ├── sample_audio.wav           # Example audio for quick testing
    └── output/
        └── transcription_output.txt   # Auto-generated transcription log
```

---

## 🛠️ Installation

### 1. Clone / navigate to the project

```bash
cd TASK2/speech_recognition_system
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `torch` (required by Whisper) may take a few minutes depending on your connection speed.

---

## 🎬 FFmpeg Setup

FFmpeg is required for reading `.mp3` and `.m4a` files, and by Whisper for all formats.

### macOS (Homebrew)

```bash
brew install ffmpeg
```

### Ubuntu / Debian

```bash
sudo apt update && sudo apt install ffmpeg
```

### Windows

1. Download the latest build from <https://ffmpeg.org/download.html>
2. Extract and add the `bin/` folder to your system `PATH`
3. Verify: `ffmpeg -version`

---

## 🚀 Usage

```bash
python speech_to_text.py
```

### Interactive session example

```
════════════════════════════════════════════════════════════
  🎙️   AI Speech Recognition System  v1.0
════════════════════════════════════════════════════════════

  Enter audio file path: sample_audio.wav

────────────────────────────────────────────────────────────
  Choose Transcription Method
────────────────────────────────────────────────────────────
  1. Google SpeechRecognition  (requires internet)
  2. Whisper – offline         (downloads model on first run)

  Enter choice [1/2]: 2
  Model size [default: base]: base
```

---

## 🎵 Supported Audio Formats

| Format | Extension | Notes |
|---|---|---|
| WAV | `.wav` | Works without FFmpeg |
| MP3 | `.mp3` | Requires FFmpeg |
| M4A | `.m4a` | Requires FFmpeg |

---

## 📄 Example Transcription Output

The file `output/transcription_output.txt` will contain entries like:

```
════════════════════════════════════════════════════════════
Transcription saved on: 2026-05-14 13:45:22
════════════════════════════════════════════════════════════
Source File     : /path/to/sample_audio.wav
Method          : OpenAI Whisper (base)
Audio Duration  : 8.4 seconds
Processing Time : 2.1 seconds
Detected Lang   : EN

Transcription:
Hello, this is a sample audio clip used to test the speech
recognition system.
```

### Console output

```
────────────────────────────────────────────────────────────
  ✅ Results
────────────────────────────────────────────────────────────
  Audio Duration  : 8.4 seconds
  Processing Time : 2.1 seconds
  Detected Lang   : EN

  📝 Transcription:

  Hello, this is a sample audio clip used to test the speech
  recognition system.

  Show timestamps? [y/N]: y

────────────────────────────────────────────────────────────
  📍 Whisper Timestamps
────────────────────────────────────────────────────────────
  [  0.00s →  8.40s]  Hello, this is a sample audio clip ...

════════════════════════════════════════════════════════════
  ✔ Saved to: output/transcription_output.txt
════════════════════════════════════════════════════════════
```

---

## ⚠️ Error Handling

| Situation | Message |
|---|---|
| File not found | `[ERROR] File not found: 'path'` |
| Unsupported format | `[ERROR] Unsupported format '.xyz'` |
| FFmpeg not installed | `[ERROR] FFmpeg not found. Install it to read .mp3 / .m4a files.` |
| Speech unclear | `[WARNING] Google Speech Recognition could not understand the audio.` |
| No internet (Google) | `[ERROR] Could not reach Google Speech API: ...` |
| Library missing | `[ERROR] Required library 'openai-whisper' is not installed.` |

---

## 🔧 Optional Enhancements

The following features can be added to extend the project:

- 🎤 **Microphone recording** – Record live audio and transcribe in real time
- 🌍 **Language selection** – Force Whisper to use a specific language with `language="fr"` etc.
- 📊 **Model comparison** – Run both methods and display results side by side
- 🖥️ **Tkinter GUI** – Desktop interface with file picker and progress bar
- 🔁 **Batch processing** – Transcribe all audio files in a directory

---

## 📜 License

This project is created for educational purposes as part of a Python internship task.
