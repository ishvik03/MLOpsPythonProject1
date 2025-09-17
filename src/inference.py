# OLD APPROACH WITHOUT THE CONFIG FILE ------


import logging

import yaml
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

config_path = "/Users/ishaangupta/PycharmProjects/MLOpsPythonProject1/config.yaml"


def load_config(path=config_path):
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
    model_dir = cfg["model"]["path"]
    task = cfg["model"]["task"]

    # Hugging Face uses -1 for CPU
    # Cuda for GPU, and the number mentioned with the GPU.
    device = 0 if cfg["model"]["device"].startswith("cuda") else -1

    logging.info(f"Loading tokenizer and model from {model_dir}")
    logging.info(f"Loading model is used for the  task {task}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    logging.info(
        f"Creating pipeline for task={task} on device={cfg['model']['device']}"
    )
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
