import logging
import os
from typing import Optional, Any

import torch
from espnet2.bin.asr_inference import Speech2Text


def load_model(device=None):
    """Load ReazonSpeech model

    Args:
      device (str): Specify "cuda" or "cpu"

    Returns:
      espnet2.bin.asr_inference.Speech2Text
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return from_pretrained(
        "https://huggingface.co/reazon-research/reazonspeech-espnet-v2",
        os.path.join(os.getcwd(), 'cache', 'esp_cache'),
        lm_weight=0,
        device=device,
    )


def from_pretrained(
        model_tag: Optional[str] = None,
        cachedir: Optional[str] = None,
        **kwargs: Optional[Any],
):
    """Build Speech2Text instance from the pretrained model.

    Args:
        model_tag (Optional[str]): Model tag of the pretrained models.
            Currently, the tags of espnet_model_zoo are supported.

    Returns:
        Speech2Text: Speech2Text instance.

    """
    if model_tag is not None:
        try:
            from espnet_model_zoo.downloader import ModelDownloader

        except ImportError:
            logging.error(
                "`espnet_model_zoo` is not installed. "
                "Please install via `pip install -U espnet_model_zoo`."
            )
            raise
        d = ModelDownloader(cachedir=cachedir)
        kwargs.update(**d.download_and_unpack(model_tag))

    return Speech2Text(**kwargs)
