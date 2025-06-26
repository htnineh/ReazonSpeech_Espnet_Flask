import nltk
import os

current_dir = os.getcwd()
nltk.data.path.append(os.path.join(current_dir, '.cache', 'nltk_data'))
os.makedirs(os.path.join(current_dir, '.cache', 'nltk_data'), exist_ok=True)
# 检查并下载 NLTK 数据
try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger', download_dir=os.path.join(current_dir, '.cache', 'nltk_data'))
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict', download_dir=os.path.join(current_dir, '.cache', 'nltk_data'))

from flask import Flask, request, jsonify
import io
import tempfile
import librosa
import numpy as np
import soundfile as sf
from reazonspeech.espnet.asr import transcribe, audio_from_path
from custom_load_model import load_model
import threading

# 设置 Hugging Face 镜像端点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 获取当前工作目录

# 设置 Hugging Face 缓存目录为当前目录下的 hf_cache 文件夹
os.environ['HF_HOME'] = os.path.join(current_dir, '.cache', 'hf_cache')
# 设置 PyTorch 缓存目录为当前目录下的 hf_cache/torch 文件夹
os.environ['TORCH_HOME'] = os.path.join(current_dir, '.cache', 'hf_cache', 'torch')
# 设置 Transformers 缓存目录为当前目录下的 hf_cache/transformers 文件夹
os.environ['TRANSFORMERS_CACHE'] = os.path.join(current_dir, '.cache', 'hf_cache', 'transformers')

os.makedirs(os.path.join(current_dir, '.cache', 'hf_cache'), exist_ok=True)

app = Flask(__name__)
# 线程锁，确保模型操作的线程安全
model_lock = threading.Lock()
# 加载语音识别模型
model = load_model(device="cpu")


def process_audio(audio_data):
    try:
        # === Step 1: Load and resample audio to 16,000 Hz ===
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)

        # === Step 2: Amplify the audio by 1.5x and clip to avoid distortion ===
        amplified_y = np.clip(y * 1.5, -1.0, 1.0)

        # === Step 3: Write amplified audio to an in-memory buffer ===
        buffer = io.BytesIO()
        sf.write(buffer, amplified_y, 16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        temp_wav_path = os.path.join(current_dir, '.cache', 'tmp')
        os.makedirs(temp_wav_path, exist_ok=True)
        # === Step 4: Save buffer to a temp WAV file for ASR model ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_wav_path) as tmp:
            tmp.write(buffer.read())
            temp_wav_path = tmp.name

        # === Step 5: Transcribe ===
        with model_lock:
            audio = audio_from_path(temp_wav_path)
            ret = transcribe(model, audio)
            os.unlink(temp_wav_path)
        return {"status": "success", "transcribed_text": ret.segments}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "未上传音频文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    audio_data = file.read()

    result = process_audio(audio_data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
