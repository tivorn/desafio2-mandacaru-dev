import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(sentence):
    doc = nlp(sentence)

    return [word.lemma_.lower().strip() for word in doc if not word.is_stop and word.is_alpha]

def spacy_vectorizer(sentence):
    tokens = spacy_tokenizer(sentence)
    tokens = " ".join(tokens)
    tokens = nlp(tokens)

    return tokens.vector

tfidf_vectorizer = TfidfVectorizer(
    tokenizer=spacy_tokenizer,
    min_df=5,
    max_df=0.8,
    ngram_range=(1,1)
)

def bag_of_words(docs_train, docs_val):
    X_train = tfidf_vectorizer.fit_transform(docs_train).astype('float32')
    X_val = tfidf_vectorizer.transform(docs_val).astype('float32')

    return X_train, X_val

def sentence2vec(docs_train, docs_val):
    X_train = np.stack([spacy_vectorizer(sentence) for sentence in docs_train])
    X_val = np.stack([spacy_vectorizer(sentence) for sentence in docs_val])

    return X_train, X_val


def glove(docs_train, docs_val, vocab_size, maxlen, return_weights=True):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(docs_train)

    X_train = tokenizer.texts_to_matrix(docs_train)
    X_val = tokenizer.texts_to_matrix(docs_val)

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)

    embeddings_index = dict()
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    if return_weights:
        return embedding_matrix

    return X_train, X_val

