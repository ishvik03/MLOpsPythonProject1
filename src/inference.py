# OLD APPROACH WITHOUT THE CONFIG FILE ------
import logging
import os
import yaml
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


# config_path = "/Users/ishaangupta/PycharmProjects/MLOpsPythonProject1/config.yaml"


def load_config(path= CONFIG_PATH):
    # Opens your config.yaml(path mentioned)  file in read mode.
    # 'f' is a file object that lets Python read the raw text inside.
    with open(path, "r") as f:
        logging.info(f"Loading config from {path}")
        # raw YAML text and parses it into a Python dictionary
        return yaml.safe_load(f)


# load_pipeline() is the function that loads
# your Hugging Face model + tokenizer
# and returns a pipeline object ready for inference.


def load_pipeline():
    cfg = load_config()
    model_dir_config = cfg["model"]["path"]
    task = cfg["model"]["task"]
    MODEL_DIR = os.path.join(BASE_DIR, model_dir_config)

    # Hugging Face uses -1 for CPU
    # Cuda for GPU, and the number mentioned with the GPU.
    device = 0 if cfg["model"]["device"].startswith("cuda") else -1

    logging.info(f"Loading tokenizer and model from {MODEL_DIR}")
    logging.info(f"Loading model is used for the  task {task}")

    if os.path.isdir(MODEL_DIR):
        # Case 1: Local model exists (e.g., on your laptop where you saved model_sharable/)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    else:
        # Case 2: Local folder missing (e.g., GitHub Actions, or colleague doesn’t have it)
        # → fallback to Hugging Face Hub using the hub_id in config.yaml
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["hub_id"])
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["hub_id"])

    return pipeline(task, model=model, tokenizer=tokenizer, device=device)


if __name__ == "__main__":
    cfg = load_config()
    text = cfg["data"]["sample_input"]
    logging.info(f"Running inference on input: {text}")
    classifier = load_pipeline()
    ans = classifier(text)
    logging.info(f"Prediction result: {ans}")
    print(ans)  # still keep print for direct user feedback

"""
# DOUBTS FOR LATER !!!
    1. Apply this to Config File location input now, 
       Why model_dir = path (worked!) and model_dir = model_sharable ( did not work!) 
    2. 

"""
