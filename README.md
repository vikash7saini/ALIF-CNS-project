
# ALIF/TASER Adversarial Audio Attack using Whisper ASR

## Overview

This project implements a simplified version of the research paper:

"ALIF: Low-Cost Adversarial Audio Attacks on Black-Box Speech Platforms Using Linguistic Features"

The system demonstrates how small perturbations added to audio signals can manipulate Automatic Speech Recognition (ASR) systems.

---

## Features

- Whisper ASR integration
- Adversarial audio generation
- Word Error Rate (WER) evaluation
- Waveform visualization
- Spectrogram analysis
- Black-box adversarial attack demonstration

---

## Technologies Used

- Python
- Whisper ASR
- PyTorch
- Librosa
- Matplotlib
- JiWER

---

## Project Workflow

Input Audio
↓
Whisper ASR
↓
Original Transcript
↓
Adversarial Perturbation
↓
Adversarial Audio
↓
Whisper ASR
↓
Modified Transcript

---

## Experimental Result

### Original Transcript

Scientists at the Sur laboratory say they have discovered a new particle.

### Adversarial Transcript

Scientists at the Surin Laboratory say they have discovered a new particle.

### Word Error Rate

WER = 0.1667

---

## Novelty Work Done

The original TASER repository required external ASR APIs and complex configurations.

In this project:

- Whisper ASR was integrated locally.
- Simplified perturbation-based adversarial attack was implemented.
- Automatic waveform and spectrogram generation were added.
- WER evaluation was included.
- The project was made lightweight and reproducible on CPU systems.

---

## Generated Outputs

- adversarial.wav
- waveform.png
- spectrogram.png

---

## How to Run

```bash
cd demo_project
python demo.py
```

---

## Repository Structure

```text
demo_project/
├── demo.py
├── adversarial.wav
├── waveform.png
└── spectrogram.png

ASRs.py
alif_otl.py
optimizer.py
README.md
.alif_ota.py
alif_ota_run.py
alif_otl.py
alif_otil_run.py
optimizer.py

```

---

## References

- ALIF/TASER Research Paper
- OpenAI Whisper
- PyTorch Documentation
