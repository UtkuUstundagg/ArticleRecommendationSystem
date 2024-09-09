import os
import torch
import numpy as np
import string
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from gensim.models import FastText
from gensim.utils import simple_preprocess
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

global model_set

dataset_path = r"C:\Users\90553\Desktop\Krapivin2009\docsutf8"


def read_dataset():
    docs = os.listdir(dataset_path)

    texts = []
    titles = []
    sciberts = []
    fasttexts = []

    tokenizer, sci_model = scibert_model()
    fast_model = fasttext_model()

    x = 1
    for doc_name in docs:
        with open(dataset_path + "\\" + doc_name, 'r', encoding='utf-8') as file:
            text = file.read()
            title = filter_title(text)
            print(str(x) + "-" + title)
            scibert = vector_user_preferences_scibert(tokenizer, sci_model, text)
            fasttext = vector_user_preferences_fasttext(text, fast_model)
            texts.append(text)
            titles.append(title)
            sciberts.append(scibert)
            fasttexts.append(fasttext)
        x += 1

    return titles, texts, sciberts, fasttexts


def setup_scibert():
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    article_text = "This is an example article about natural language processing."
    user_profile = "I am interested in machine learning and artificial intelligence."

    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    article_embedding = outputs.pooler_output.numpy()

    inputs = tokenizer(user_profile, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    user_embedding = outputs.pooler_output.numpy()


def setup_fasttext():
    model_path = r"D:\python-fasttextmodel\wiki.en.bin"
    model = FastText.load_fasttext_format(model_path)

    article_text = "This is an example article about natural language processing."
    user_profile = "I am interested in machine learning and artificial intelligence."

    article_tokens = simple_preprocess(article_text)
    article_embedding = np.mean([model.wv[word] for word in article_tokens if word in model.wv], axis=0)

    user_tokens = simple_preprocess(user_profile)
    user_embedding = np.mean([model.wv[word] for word in user_tokens if word in model.wv], axis=0)


def scibert_model():
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model


def fasttext_model():
    model_path = r"D:\python-fasttextmodel\wiki.en.bin"
    model = FastText.load_fasttext_format(model_path)

    return model


def filter_title(text):
    first_index = text.find("--T")
    if first_index != -1:
        last_index = text.find("--A", first_index)
        if last_index != -1:
            title = text[first_index + len("--T"):last_index]
            return title.strip()
        else:
            return None
    else:
        return None


def preprocess_preferences(preferences):
    preferences = preferences.lower()
    preferences_without_punctuation = preferences.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(preferences_without_punctuation)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    return stemmed_tokens


def vector_user_preferences_scibert(tokenizer, model, preferences):
    tokens = preprocess_preferences(preferences)

    inputs = tokenizer(tokens, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    user_embeddings = outputs.pooler_output.numpy()
    user_embedding = np.mean(user_embeddings, axis=0)

    return user_embedding


def vector_user_preferences_fasttext(preferences, model):
    tokens = preprocess_preferences(preferences)

    user_embedding = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
    return user_embedding


def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


def calc_recommendations(user_vector, article_vectors, top_n=5):
    similarities = [(i, compute_cosine_similarity(user_vector, article_vector)) for i, article_vector in
                    enumerate(article_vectors)]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def get_article_titles(indices, articles):
    titles = []
    for index in indices:
        title = articles[index]['Title']
        titles.append(title)

    return titles


def get_recommendation(user, articles):
    user_fasttext_vector = user['fasttext_vector']
    user_scibert_vector = user['scibert_vector']

    article_fasttext_vectors = [article['Fasttext_V'] for article in articles]
    article_scibert_vectors = [article['Scibert_V'] for article in articles]

    fasttext_recommendations = calc_recommendations(user_fasttext_vector, article_fasttext_vectors)
    fasttext_indices = [rec[0] for rec in fasttext_recommendations]
    fasttext_titles = get_article_titles(fasttext_indices, articles)

    scibert_recommendations = calc_recommendations(user_scibert_vector, article_scibert_vectors)
    scibert_indices = [rec[0] for rec in scibert_recommendations]
    scibert_titles = get_article_titles(scibert_indices, articles)

    return scibert_titles, fasttext_titles

