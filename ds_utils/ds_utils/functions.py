import numpy as np


def vectorize(list_of_docs, model, strategy):
    features = []
    size_output = model.vector_size
    if strategy == "min-max":
        size_output *= 2

    for tokens in list_of_docs:
        zero_vector = np.zeros(size_output)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
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