from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Initialize tokenizer and model for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = TFAutoModel.from_pretrained("distilbert-base-multilingual-cased")

def compute_embedding(text):
    encoded_input = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**encoded_input)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()


# Load a subset of the wikipedia dataset (assuming structure and availability)
dataset = load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train", streaming=True)


# ========Exercise 3.1 ===========
# Fill in the following code
# ===============================
def find_most_relevant_article(query_embedding, dataset, max_num_of_articles=None):
    most_relevant_article = None
    max_similarity = -1  # Setting to -1 in case the values are negative

    for row in tqdm(dataset, total=max_num_of_articles):
        if max_num_of_articles is not None and row['id'] == max_num_of_articles:
            break

        article_text_embedding = compute_embedding(row['text'])
        text_sim = cosine_similarity(query_embedding, article_text_embedding).flatten()[0]

        if text_sim > max_similarity:
            max_similarity = text_sim
            most_relevant_article = row['title']

    return most_relevant_article, max_similarity
# ========End Exercise 3.1 ===========


input_text = ["Leonardo DiCaprio", "France", "Python", "Deep Learning"]

for text in input_text:
    # Compute the embedding for the input text
    input_embedding = compute_embedding(text)

    # Find the most relevant article
    # To reduce the runtime, look at only the first N articles
    article, similarity = find_most_relevant_article(input_embedding, dataset, 1000)
    print("Input Text: ", text)
    print("Most Relevant Article: ", article)
    print("Similarity Score: ", similarity)
