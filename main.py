import matplotlib.pyplot as plt
from datasets import load_dataset

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, T5ForConditionalGeneration

# Download necessary resources from nltk
# nltk.download('punkt')
# nltk.download('stopwords')

dataset = load_dataset("cnn_dailymail", '3.0.0')
df = dataset['train'].to_pandas()
df = df.head(1000)

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Convert to lower case
    tokens = [token.lower() for token in tokens]
    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)

###====== Part 2.1 =====================
###Write a code that creates two new columns -  artice_len and highlights_len


df['article_len'] = [len(df['article'][article]) for article in range(len(df['article']))]
df['highlights_len'] = [len(df['highlights'][highlight]) for highlight in range(len(df['highlights']))]


###====== Part 2.2 =====================
### Fill in this code
def plot_histograms(df):
    plt.subplot(1, 2, 1)
    plt.hist(df['article_len'])
    plt.title('article_len')

    plt.subplot(1, 2, 2)
    plt.hist(df['highlights_len'])
    plt.title('highlights_len')

    plt.suptitle('Length Histograms')
    plt.show()
    return None


# plot_histograms(df)


###======Part 2.3 ================
### Fill in the code
def ngrams(text, n):
    # Preprocess the text first
    processed_text = preprocess_text(text)
    words = processed_text.split()
    return set(zip(*[words[i:] for i in range(n)]))


def rouge_n(reference, candidate, n):
    ref_ngrams = ngrams(reference, n)
    cand_ngrams = ngrams(candidate, n)

    ref_count = len(ref_ngrams)
    cand_count = len(cand_ngrams)

    overlap_count = sum(1 for ngram in cand_ngrams if ngram in ref_ngrams)

    recall = overlap_count / ref_count if ref_count > 0 else 0.0
    precision = overlap_count / cand_count if cand_count > 0 else 0.0
    f1_score = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0

    return f1_score


###=========== 2.3 ================

# Example of calculating Rouge-1 and Rouge-2 for a dataframe
# df['rouge_1'] = df.apply(lambda row: rouge_n(row['highlights'], row['article'], 1), axis=1)
# df['rouge_2'] = df.apply(lambda row: rouge_n(row['highlights'], row['article'], 2), axis=1)

# plt.figure(figsize=(12, 6))
# plt.hist(df['rouge_2'], bins=30, color='blue', alpha=0.7)
# plt.title('Rouge-2 score distribution on ground truth')
# plt.show()

# min_rouge_1 = df['rouge_1'].min()
# print("Lowest Rouge-1 score: ", min_rouge_1)
# max_rouge_1 = df['rouge_1'].max()
# print("Highest Rouge-1 score: ", max_rouge_1)
# min_rouge_2 = df['rouge_2'].min()
# print("Lowest Rouge-2 score: ", min_rouge_2)
# max_rouge_2 = df['rouge_2'].max()
# print("Highest Rouge-2 score: ", max_rouge_2)
#
# min_rouge_2_index = df['rouge_2'].argmin()
# lowest_r2_article = df.iloc[min_rouge_2_index]['article']
# print("Index of article with lowest Rouge-2 score:", min_rouge_2_index)
# print("========================")
# print("Article with lowest Rouge-2 score:", df.iloc[min_rouge_2_index]['article'])
# print("========================")
# print("Highlights with lowest Rouge-2 score:", df.iloc[min_rouge_2_index]['highlights'])
# print("========================")
# print("Our summary of the article with the lowest score:", lowest_summary)


###=========== 2.4 ================
# Initialize the summarization pipeline
df = dataset['train'].select(range(10))
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')
summarizer = T5ForConditionalGeneration.from_pretrained('t5-small')


# Define the preprocessing function
def summarize_text(text):
    # Preprocess the input text
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary using the T5 model
    summary_ids = summarizer.generate(inputs.input_ids, max_length=20, min_length=5, do_sample=False)

    # Decode and return the summary text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# Generate summaries
generated_summaries = [summarize_text(article) for article in df['article']]
df = df.add_column("generated_summaries", generated_summaries)

#Calculate the rouge-2 score of the first 10 entries
df = df.map(lambda row: {"rouge_2_generated": rouge_n(row["generated_summaries"], row["article"], 2)})

# Calculate ROUGE-2 scores for the original 'highlights' against the 'article' (ground truth)
df = df.map(lambda row: {"rouge_2_ground_truth": rouge_n(row["highlights"], row["article"], 2)})

for rouge2_ in df['rouge_2_generated']:
    print(rouge2_)

# for rouge2_gen, rouge_2_gt in zip(df['rouge_2_generated'], df['rouge_2_ground_truth']):
#     print(rouge2_gen < rouge_2_gt)
