# 🧠 NLP Toolkit: Text Summarization & Information Retrieval

This project demonstrates two core applications of modern Natural Language Processing (NLP):

1. **Abstractive Text Summarization** using a pretrained T5 model.
2. **Semantic Information Retrieval** using multilingual DistilBERT embeddings.

---

## 📂 Project Structure

### 🔹 `text_summarization.py`

This module:
- Loads 1000 articles from the **CNN/DailyMail** dataset.
- Computes the length of articles and summaries.
- Calculates **ROUGE-1** and **ROUGE-2** scores between:
  - The article and the ground truth summary (human-written)
  - The article and a generated summary from `t5-small`
- Visualizes distribution of lengths and ROUGE scores.
- Identifies cases with lowest summarization quality.

### 🔹 `information_retrieval.py`

This module:
- Embeds input queries using **DistilBERT** (multilingual, via Hugging Face Transformers).
- Loads a streaming subset of the **Wikipedia 2022** dataset (via Hugging Face).
- Computes **cosine similarity** between a query and Wikipedia article embeddings.
- Finds the most semantically relevant article titles for each query input (e.g. `"Python"` → `"Python (programming language)"`).

---

## 🛠️ Technologies Used

- 🧠 Hugging Face Transformers (`t5-small`, `distilbert-base-multilingual-cased`)
- 📊 ROUGE metric (custom implementation)
- 📚 Datasets: `cnn_dailymail`, `Cohere/wikipedia-22-12-en-embeddings`
- 📈 Visualization: Matplotlib
- 🧪 Frameworks: TensorFlow, Scikit-learn
- ⚙️ Tokenization and Embeddings: `AutoTokenizer`, `TFAutoModel`
- 📝 Text preprocessing: `nltk`

---

## 🚀 Getting Started

### Install dependencies:
```bash
pip install transformers datasets scikit-learn nltk tensorflow tqdm matplotlib
```

### Optional (first run only):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run summarization:
```bash
python text_summarization.py
```

### Run information retrieval:
```bash
python information_retrieval.py
```

---

## 📌 Notes

- The summarization module currently runs on a small subset of the data for speed. You can scale it up easily.
- The Wikipedia dataset is streamed to avoid memory issues — processing is intentionally capped for fast demo.

---

## ✨ Future Ideas

- Integrate both modules into a web app using Streamlit or Gradio.
- Support question answering or hybrid search using retrieved articles.
- Add BLEU or METEOR metrics to complement ROUGE.
- Train/fine-tune models on domain-specific data.
