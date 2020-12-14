import re
import string
from pathlib import Path

import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

nltk.download("stopwords")

ROOT_PATH = Path(__file__).resolve().parents[1]
NEWS_DATA = ROOT_PATH / "data" / "news_data.csv"
STOPWORDS = set(stopwords.words("english") + ["news", "new", "top"])


class Tokenizer:
    def __init__(self, stopwords=STOPWORDS, tokenizer=word_tokenize):
        """Tokenizer class to create custom processing pipeline.

        Args:
            stopwords: List of stop-words.
            tokenizer: Tokenizer.
        """
        self._stopwords = stopwords
        self._tokenizer = tokenizer

    def __call__(self, doc):
        """Call method to use as a sklearn custom analyzer"""
        tokens = self._tokenize(doc)
        return [token for token in tokens]

    def _tokenize(self, text):
        """Pre-process text and generate tokens.

        Args:
            text: Text to tokenize.

        Returns:
            Tokenized text.
        """
        text = str(text).lower()  # Lowercase words
        text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
        text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
        text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
        text = re.sub(
            f"[{re.escape(string.punctuation)}]", "", text
        )  # Remove punctuation

        tokens = self._tokenizer(text)  # Get tokens from text
        tokens = [t for t in tokens if not t in self._stopwords]  # Remove stopwords
        tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
        tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
        return tokens


def load_data(data="news"):
    if data == "news":
        return pd.read_csv(NEWS_DATA)
    else:
        raise ValueError(f"Dataset {data} not found!")


def clean_news_data(df_orig):

    tokenizer = Tokenizer()
    text_columns = ["title", "description", "content"]

    df = df_orig.copy()
    df["content"] = df["content"].fillna("")

    for col in text_columns:
        df[col] = df[col].astype(str)

    # Create text column based on title, description, and content
    df["text"] = df[text_columns].apply(lambda x: " | ".join(x), axis=1)
    df["tokens"] = df["text"].map(lambda x: tokenizer(x))

    # Remove duplicated after preprocessing
    _, idx = np.unique(df["tokens"], return_index=True)
    df = df.iloc[idx, :]

    # Remove empty values
    df = df.loc[df.tokens.map(lambda x: len(x) > 0), ["text", "tokens"]]

    print(f"Original dataframe: {df_orig.shape}")
    print(f"Pre-processed dataframe: {df.shape}")
    return df


def vectorize(list_of_docs, model, strategy):
    """Generate vectors for list of documents using a Word Embedding.

    Args:
        list_of_docs: List of documents.
        model: Gensim Word Embedding.
        strategy: Aggregation strategy ("average", or "min-max".)

    Raises:
        ValueError: If the strategy is other than "average" or "min-max".

    Returns:
        List of vectors.
    """
    features = []
    size_output = model.vector_size
    embedding_dict = model

    if strategy == "min-max":
        size_output *= 2

    if hasattr(model, "wv"):
        embedding_dict = model.wv

    for tokens in list_of_docs:
        zero_vector = np.zeros(size_output)
        vectors = []
        for token in tokens:
            if token in embedding_dict:
                try:
                    vectors.append(embedding_dict[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            if strategy == "min-max":
                min_vec = vectors.min(axis=0)
                max_vec = vectors.max(axis=0)
                features.append(np.concatenate((min_vec, max_vec)))
            elif strategy == "average":
                avg_vec = vectors.mean(axis=0)
                features.append(avg_vec)
            else:
                raise ValueError(f"Aggregation strategy {strategy} does not exist!")
        else:
            features.append(zero_vector)
    return features


def mbkmeans_clusters(X, k, mb=500, print_silhouette_values=False):
    """Generate clusters.

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches. Defaults to 500.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_
