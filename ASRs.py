import whisper
import os

model = whisper.load_model("base")

def transcribe_audio(srcdir, file):
    path = os.path.join(srcdir, file)
    result = model.transcribe(path)
    return result["text"]

def get_trans_tencent(srcdir, file):
    return transcribe_audio(srcdir, file)

def get_trans_iflytek(srcdir, file):
    return transcribe_audio(srcdir, file)

def get_trans_amazon(srcdir, file):
    return transcribe_audio(srcdir, file)

def get_trans_google(srcdir, file):
    return transcribe_audio(srcdir, file)

def get_trans_azure(srcdir, file):
    return transcribe_audio(srcdir, file)
