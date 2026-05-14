"""
AI Speech Recognition System
==============================
Converts short audio clips into text using pre-trained speech-to-text models.

Supported Methods:
    1. Google Web Speech API (via SpeechRecognition library) - requires internet
    2. OpenAI Whisper (offline / local) - downloads model on first use

Supported Formats: .wav, .mp3, .m4a

Author  : AI Speech Recognition System
Version : 1.0.0
"""

import os
import sys
import time
import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Helper: check for third-party libraries gracefully
# ──────────────────────────────────────────────────────────────────────────────

def _require(module_name: str, pip_name: str):
    """Import a module and raise a friendly error if it is not installed."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"\n[ERROR] Required library '{pip_name}' is not installed.")
        print(f"        Run:  pip install {pip_name}\n")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Supported audio formats
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a"}


# ──────────────────────────────────────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_audio_duration(audio_path: str) -> float:
    """
    Return the duration of an audio file in seconds.

    Uses pydub's AudioSegment, which supports .wav, .mp3, and .m4a
    (requires FFmpeg to be installed for non-WAV formats).

    Parameters
    ----------
    audio_path : str
        Absolute or relative path to the audio file.

    Returns
    -------
    float
        Duration in seconds, or 0.0 if the file cannot be read.
    """
    pydub = _require("pydub", "pydub")
    AudioSegment = pydub.AudioSegment

    ext = os.path.splitext(audio_path)[1].lower()
    try:
        if ext == ".wav":
            audio = AudioSegment.from_wav(audio_path)
        elif ext == ".mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif ext == ".m4a":
            audio = AudioSegment.from_file(audio_path, format="m4a")
        else:
            print(f"[WARNING] Unsupported format '{ext}' for duration check.")
            return 0.0

        duration_seconds = len(audio) / 1000.0  # pydub works in milliseconds
        return round(duration_seconds, 2)

    except FileNotFoundError:
        print(f"[ERROR] FFmpeg not found. Install it to read .mp3 / .m4a files.")
        print("        macOS  : brew install ffmpeg")
        print("        Ubuntu : sudo apt install ffmpeg")
        print("        Windows: https://ffmpeg.org/download.html")
        return 0.0
    except Exception as exc:
        print(f"[ERROR] Could not determine audio duration: {exc}")
        return 0.0


def _convert_to_wav(audio_path: str) -> str:
    """
    Convert mp3 / m4a to a temporary WAV file for SpeechRecognition.

    Parameters
    ----------
    audio_path : str
        Path to the original audio file.

    Returns
    -------
    str
        Path to the converted (temporary) WAV file, or the original path
        if conversion is not required.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path  # already WAV – no conversion needed

    pydub = _require("pydub", "pydub")
    AudioSegment = pydub.AudioSegment

    tmp_wav = audio_path.rsplit(".", 1)[0] + "_tmp_converted.wav"
    try:
        if ext == ".mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif ext == ".m4a":
            audio = AudioSegment.from_file(audio_path, format="m4a")
        else:
            return audio_path

        audio.export(tmp_wav, format="wav")
        return tmp_wav
    except Exception as exc:
        print(f"[ERROR] Conversion to WAV failed: {exc}")
        return audio_path


def transcribe_with_speechrecognition(audio_path: str) -> str:
    """
    Transcribe an audio file using the SpeechRecognition library
    with Google's free Web Speech API.

    Requirements
    ------------
    - Active internet connection
    - 'speechrecognition' Python package  (pip install speechrecognition)
    - FFmpeg for .mp3 / .m4a input files

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav, .mp3, or .m4a).

    Returns
    -------
    str
        Transcribed text, or an empty string if transcription failed.
    """
    sr = _require("speech_recognition", "speechrecognition")
    recognizer = sr.Recognizer()

    # Convert to WAV if necessary
    wav_path = _convert_to_wav(audio_path)
    tmp_created = (wav_path != audio_path)

    try:
        with sr.AudioFile(wav_path) as source:
            print("  [INFO] Reading audio file…")
            # Adjust for ambient noise to improve accuracy
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        print("  [INFO] Sending audio to Google Web Speech API…")
        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        print("[WARNING] Google Speech Recognition could not understand the audio.")
        return ""
    except sr.RequestError as exc:
        print(f"[ERROR] Could not reach Google Speech API: {exc}")
        print("        Please check your internet connection.")
        return ""
    except Exception as exc:
        print(f"[ERROR] SpeechRecognition error: {exc}")
        return ""
    finally:
        # Clean up the temporary WAV file if one was created
        if tmp_created and os.path.exists(wav_path):
            os.remove(wav_path)


def transcribe_with_whisper(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe an audio file using OpenAI Whisper (fully offline).

    On first run, Whisper automatically downloads the requested model
    (~150 MB for 'base').  Subsequent runs use the cached model.

    Available model sizes (speed vs. accuracy tradeoff):
        tiny | base | small | medium | large

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav, .mp3, or .m4a).
    model_size : str, optional
        Whisper model variant to use (default: "base").

    Returns
    -------
    dict
        {
            "text"      : str,           # full transcription
            "segments"  : list[dict],    # per-segment data with timestamps
            "language"  : str            # detected language code
        }
        Returns {"text": "", "segments": [], "language": ""} on failure.
    """
    whisper = _require("whisper", "openai-whisper")

    VALID_SIZES = {"tiny", "base", "small", "medium", "large"}
    if model_size not in VALID_SIZES:
        print(f"[WARNING] Unknown model size '{model_size}'. Falling back to 'base'.")
        model_size = "base"

    try:
        print(f"  [INFO] Loading Whisper '{model_size}' model…")
        print("         (First run will download the model — please wait)")
        model = whisper.load_model(model_size)

        print("  [INFO] Transcribing with Whisper…")
        result = model.transcribe(audio_path)

        return {
            "text"    : result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
        }

    except FileNotFoundError:
        print("[ERROR] FFmpeg not found.  Whisper requires FFmpeg to decode audio.")
        print("        macOS  : brew install ffmpeg")
        print("        Ubuntu : sudo apt install ffmpeg")
        print("        Windows: https://ffmpeg.org/download.html")
        return {"text": "", "segments": [], "language": ""}
    except Exception as exc:
        print(f"[ERROR] Whisper transcription failed: {exc}")
        return {"text": "", "segments": [], "language": ""}


def save_transcription(text: str, output_path: str) -> bool:
    """
    Save a transcription string to a text file.

    Creates parent directories automatically if they do not exist.

    Parameters
    ----------
    text : str
        The transcription text to save.
    output_path : str
        Destination file path (e.g. 'output/transcription_output.txt').

    Returns
    -------
    bool
        True if the file was saved successfully, False otherwise.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_path, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Transcription saved on: {timestamp}\n")
            f.write("=" * 60 + "\n")
            f.write(text + "\n\n")

        return True
    except PermissionError:
        print(f"[ERROR] Permission denied when writing to '{output_path}'.")
        return False
    except Exception as exc:
        print(f"[ERROR] Could not save transcription: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner():
    """Print a styled banner."""
    print("\n" + "═" * 60)
    print("  🎙️   AI Speech Recognition System  v1.0")
    print("═" * 60)


def _section(title: str):
    """Print a section divider."""
    print("\n" + "─" * 60)
    print(f"  {title}")
    print("─" * 60)


def _print_timestamps(segments: list):
    """Print Whisper segment-level timestamps."""
    if not segments:
        return
    _section("📍 Whisper Timestamps")
    for seg in segments:
        start = seg.get("start", 0)
        end   = seg.get("end", 0)
        seg_text = seg.get("text", "").strip()
        print(f"  [{start:6.2f}s → {end:6.2f}s]  {seg_text}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Program
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Interactive CLI for the AI Speech Recognition System.

    Flow
    ----
    1. Ask the user for the audio file path.
    2. Validate file existence and format.
    3. Show method menu.
    4. Transcribe and display results.
    5. Save to output/transcription_output.txt.
    """
    _banner()

    # ── Step 1: Get audio file path ──────────────────────────────────────────
    print()
    audio_path = input("  Enter audio file path: ").strip().strip('"').strip("'")

    # ── Step 2: Validate the file ────────────────────────────────────────────
    if not audio_path:
        print("[ERROR] No file path provided. Exiting.")
        sys.exit(1)

    if not os.path.isfile(audio_path):
        print(f"[ERROR] File not found: '{audio_path}'")
        print("        Please check the path and try again.")
        sys.exit(1)

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        print(f"[ERROR] Unsupported format '{ext}'.")
        print(f"        Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        sys.exit(1)

    # ── Step 3: Show menu ────────────────────────────────────────────────────
    _section("Choose Transcription Method")
    print("  1. Google SpeechRecognition  (requires internet)")
    print("  2. Whisper – offline         (downloads model on first run)")
    print()

    choice = input("  Enter choice [1/2]: ").strip()
    if choice not in {"1", "2"}:
        print("[ERROR] Invalid choice. Please enter 1 or 2.")
        sys.exit(1)

    # ── Step 4: Get audio duration ───────────────────────────────────────────
    _section("📊 Audio Info")
    print("  Calculating audio duration…")
    duration = get_audio_duration(audio_path)
    if duration > 0:
        print(f"  Audio Duration : {duration} seconds")
    else:
        print("  Audio Duration : (could not determine)")

    # ── Step 5: Transcribe ───────────────────────────────────────────────────
    _section("🔄 Transcribing…")
    start_time = time.time()
    transcription = ""
    whisper_result = None

    if choice == "1":
        print("  Method: Google Web Speech API\n")
        transcription = transcribe_with_speechrecognition(audio_path)

    else:
        # Whisper – optionally ask for model size
        print("  Method: OpenAI Whisper (offline)\n")
        print("  Available model sizes: tiny | base | small | medium | large")
        model_size = input("  Model size [default: base]: ").strip().lower()
        if not model_size:
            model_size = "base"
        whisper_result = transcribe_with_whisper(audio_path, model_size=model_size)
        transcription = whisper_result["text"]

    processing_time = round(time.time() - start_time, 2)

    # ── Step 6: Display results ──────────────────────────────────────────────
    _section("✅ Results")
    print(f"  Audio Duration  : {duration} seconds")
    print(f"  Processing Time : {processing_time} seconds")

    if whisper_result:
        lang = whisper_result.get("language", "unknown")
        print(f"  Detected Lang   : {lang.upper()}")

    print()
    if transcription:
        print("  📝 Transcription:")
        print()
        # Word-wrap at ~70 chars for readability
        words = transcription.split()
        line, lines = "", []
        for word in words:
            if len(line) + len(word) + 1 > 70:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"  {l}")
    else:
        print("  [!] No transcription produced.")
        print("      • Check that the audio contains clear speech.")
        print("      • If using Google API, verify your internet connection.")

    # Print Whisper timestamps if available
    if whisper_result and whisper_result.get("segments"):
        show_ts = input("\n  Show timestamps? [y/N]: ").strip().lower()
        if show_ts == "y":
            _print_timestamps(whisper_result["segments"])

    # ── Step 7: Save output ──────────────────────────────────────────────────
    output_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    output_path = os.path.join(output_dir, "transcription_output.txt")

    if transcription:
        _section("💾 Saving Transcription")
        method_label = (
            "Google SpeechRecognition"
            if choice == "1"
            else f"OpenAI Whisper ({model_size if choice == '2' else 'base'})"
        )
        full_text = (
            f"Source File     : {os.path.abspath(audio_path)}\n"
            f"Method          : {method_label}\n"
            f"Audio Duration  : {duration} seconds\n"
            f"Processing Time : {processing_time} seconds\n"
            + (f"Detected Lang   : {whisper_result['language'].upper()}\n"
               if whisper_result else "")
            + f"\nTranscription:\n{transcription}\n"
        )
        success = save_transcription(full_text, output_path)
        if success:
            print(f"  ✔ Saved to: {output_path}")
        else:
            print("  ✘ Failed to save transcription.")
    else:
        print("\n  [INFO] Nothing to save (empty transcription).")

    print("\n" + "═" * 60)
    print("  Thank you for using AI Speech Recognition System!")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
