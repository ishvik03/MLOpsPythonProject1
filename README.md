## ⚡ Model Setup  

The model files are **not included** in this repository (ignored via `.gitignore`).  
To run the project, download a Hugging Face model (example: `distilbert-base-uncased-finetuned-sst-2-english`).  

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### 📂 Config File
Update config.yaml with your model details:
model:
  path: "distilbert-base-uncased-finetuned-sst-2-english"
  task: "sentiment-analysis"
  device: "cpu"

data:
  sample_input: "I love learning MLOps!"

▶️ Run
python src/inference/inference.py
