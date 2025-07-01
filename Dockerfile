FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL authors="gxf99"

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    make \
    python3-pip \
    python3-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    nltk \
    librosa \
    numpy \
    soundfile \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/reazon-research/ReazonSpeech.git \
    && pip install ReazonSpeech/pkg/espnet-asr\
    && rm -rf ReazonSpeech

RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && pip install --no-cache-dir flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && rm flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


COPY main.py custom_load_model.py ./

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

ENTRYPOINT ["gunicorn", "-w", "1","--timeout","0", "-b", "0.0.0.0:5000", "main:app"]
