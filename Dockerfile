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
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    nltk \
    librosa \
    numpy \
    soundfile \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN git clone https://github.com/reazon-research/ReazonSpeech.git \
    && pip install --no-cache-dir ReazonSpeech/pkg/espnet-asr \
#    && rm -rf ReazonSpeech




COPY main.py custom_load_model.py ./

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

ENTRYPOINT ["gunicorn", "-w", "1","--timeout","0", "-b", "0.0.0.0:5000", "main:app"]
