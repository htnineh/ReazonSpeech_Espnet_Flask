FROM python:3.10
LABEL authors="gxf99"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    && git clone https://github.com/reazon-research/ReazonSpeech.git \
    && pip install ReazonSpeech/pkg/espnet-asr \
    && rm -rf ReazonSpeech

COPY main.py custom_load_model.py ./

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

ENTRYPOINT ["gunicorn", "-w", "1","--timeout","0", "-b", "0.0.0.0:5000", "main:app"]
