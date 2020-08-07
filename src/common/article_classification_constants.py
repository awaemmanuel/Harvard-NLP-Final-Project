from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

MAX_SEQ_LEN = 512


SUPPORTED_MODELS = [
    list(x.pretrained_config_archive_map)
    for x in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
]
SUPPORTED_MODELS = sorted([x for y in SUPPORTED_MODELS for x in y])

MIND_DATA_MAP = {
    "training_small": "MINDsmall_train.zip",
    "training_large": "MINDlarge_train.zip",
    "validation_small": "MINDsmall_dev.zip",
    "validation_large": "MINDlarge_dev.zip",
}

MIND_DATA_URL = "https://mind201910small.blob.core.windows.net/release"
