FROM python:3.10
LABEL authors="gxf99"

ENTRYPOINT ["top", "-b"]

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    && git clone https://github.com/reazon-research/ReazonSpeech.git \
    && pip install --no-cache-dir ReazonSpeech/pkg/nemo-asr \
    && rm -rf ReazonSpeech

COPY main.py custom_load_model.py ./

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
