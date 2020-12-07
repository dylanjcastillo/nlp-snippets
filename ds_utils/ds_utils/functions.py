import string
import re
import numpy as np


def vectorize(list_of_docs, model, strategy):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim Word Embedding
        strategy: Aggregation strategy ("average", or "min-max")

    Raises:
        ValueError: If the strategy is other than "average" or "min-max"

    Returns:
        List of vectors
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


def generate_clusters(X, k, mb=500, random_state=42):
    """Generate clusters

    Args:
        X: Matrix of features
        k: Number of clusters
        mb: Size of mini-batches. Defaults to 500.
        random_state: Random seed. Defaults to 42.

    Returns:
        Trained clustering model and labels based on X
    """
    clustering = MiniBatchKMeans(n_clusters=k, batch_size=mb, random_state=random_state)
    cluster_labels = clustering.fit_predict(X)
    print(f"For n_clusters = {k}")
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"The average Silhouette_score is: {silhouette_avg:.2f}")
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        print(
            f"    Silhoute values for cluster {i}: "
            f"Size:{ith_cluster_silhouette_values.shape[0]}"
            f"| Min:{ith_cluster_silhouette_values.min():.2f}"
            f"| Avg:{ith_cluster_silhouette_values.mean():.2f}"
            f"| Max: {ith_cluster_silhouette_values.max():.2f}"
        )
    try:
        print(f"The Inertia is :{clustering.inertia_}")
        distorsions.append(clustering.inertia_)
    except:
        pass
    return clustering, cluster_labels


def generate_tokens(
    text,
    stop_words,
    tokenizer,
):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize
        tokenizer: Tokenizer
        stop_words: List of stop-words

    Returns:
        Tokenized text
    """
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stop_words]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens