import certifi
from bson import ObjectId
from pymongo import MongoClient

from main import vector_user_preferences_scibert, vector_user_preferences_fasttext, scibert_model, fasttext_model, \
    get_recommendation

db_username = "utku70698"
db_password = "n5fopfq40LrUYJ7V"

connection = ('mongodb+srv://utku70698:n5fopfq40LrUYJ7V@cluster0.khpgy25.mongodb.net/?retryWrites=true&w=majority'
              '&appName=Cluster0')


def import_dataset(titles, texts, scibert, fasttext):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    articles_collection = db["articles"]

    for title, text, scibert_vector, fasttext_vector in zip(titles, texts, scibert, fasttext):
        data = {"Title": title, "Text": text, "Scibert_V": scibert_vector.tolist(), "Fasttext_V": fasttext_vector.tolist()}
        articles_collection.insert_one(data)

    client.close()


def find_all_articles():
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    articles_collection = db["articles"]

    articles = articles_collection.find()
    results_list = []

    for result in articles:
        results_list.append(result)

    client.close()
    return results_list


def find_one_article(keyword):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    articles_collection = db["articles"]

    article = articles_collection.find_one({'Title': {'$regex': keyword, '$options': 'i'}})

    client.close()
    return article


def add_user(username, password, preferences):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    users_collection = db["users"]

    data = {"username": username, "password": password, "preferences": preferences}
    insertion_id = users_collection.insert_one(data)
    client.close()

    vectorize_user(insertion_id)


def find_user_by_username_password(username, password):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    users_collection = db["users"]

    user = users_collection.find_one({'username': username, 'password': password})
    client.close()

    if user:
        return user
    else:
        return None


def find_user_by_id(user_id):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    users_collection = db["users"]

    user_id = ObjectId(user_id)
    user = users_collection.find_one({'_id': user_id})
    client.close()

    return user


def update_user_interests(user_id, preferences):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    users_collection = db["users"]

    user = find_user_by_id(user_id)
    user['preferences'] = preferences

    filter_criteria = {'_id': ObjectId(user['_id'])}
    update_data = {'$set': {'preferences': str(preferences)}}

    users_collection.update_one(filter_criteria, update_data)
    client.close()

    vectorize_user(user_id)


def update_user_vector(user_id, scibert, fasttext):
    client = MongoClient(connection, tlsCAFile=certifi.where())
    db = client["db1"]
    users_collection = db["users"]

    user = find_user_by_id(user_id)
    user['scibert_vector'] = scibert
    user['fasttext_vector'] = fasttext

    filter_criteria = {'_id': ObjectId(user['_id'])}
    update_data = {'$set': {'scibert_vector': scibert.tolist(), 'fasttext_vector': fasttext.tolist()}}

    users_collection.update_one(filter_criteria, update_data)
    client.close()


def vectorize_user(user_id):
    preferences = find_user_by_id(user_id)['preferences']

    tokenizer, sci_model = scibert_model()
    fast_model = fasttext_model()

    user_vector_scibert = vector_user_preferences_scibert(tokenizer, sci_model, preferences)
    user_vector_fasttext = vector_user_preferences_fasttext(preferences, fast_model)

    update_user_vector(user_id, user_vector_scibert, user_vector_fasttext)


# find_user_by_username_password("utku", "123")
# vectorize_user("6641d3fa032d359b1b6594b7")


