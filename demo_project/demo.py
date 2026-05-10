import whisper

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from jiwer import wer

# Load Whisper model
model = whisper.load_model("base")

# Input audio
input_audio = "../tacotron2/demo.wav"

# -----------------------------------
# Original Transcription
# -----------------------------------

result = model.transcribe(input_audio)
original_text = result["text"]

print("\nOriginal Transcript:")
print(original_text)

# -----------------------------------
# Load Audio
# -----------------------------------

audio, sr = sf.read(input_audio)

# -----------------------------------
# Generate Adversarial Perturbation
# -----------------------------------

noise = np.random.normal(0, 0.01, audio.shape)

adv_audio = audio + noise

# Save adversarial audio
sf.write("adversarial.wav", adv_audio, sr)

# -----------------------------------
# Adversarial Transcription
# -----------------------------------

result2 = model.transcribe("adversarial.wav")
adv_text = result2["text"]

print("\nAdversarial Transcript:")
print(adv_text)

# -----------------------------------
# Word Error Rate
# -----------------------------------

score = wer(original_text, adv_text)

print("\nWER Score:")
print(score)

# -----------------------------------
# Waveform Comparison Graph
# -----------------------------------

plt.figure(figsize=(12, 4))

plt.plot(audio[:5000], label="Original")
plt.plot(adv_audio[:5000], alpha=0.7, label="Adversarial")

plt.legend()

plt.title("Waveform Comparison")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig("waveform.png")

# -----------------------------------
# Spectrogram Graph
# -----------------------------------

plt.figure(figsize=(12, 4))

S = librosa.amplitude_to_db(
    np.abs(librosa.stft(adv_audio)),
    ref=np.max
)

librosa.display.specshow(
    S,
    sr=sr,
    x_axis='time',
    y_axis='log'
)

plt.colorbar(format='%+2.0f dB')

plt.title("Adversarial Audio Spectrogram")

plt.tight_layout()

plt.savefig("spectrogram.png")

print("\nGenerated Files:")
print("1. adversarial.wav")
print("2. waveform.png")
print("3. spectrogram.png")
