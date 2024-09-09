from flask import Blueprint, render_template, request, redirect, session, jsonify

from database import add_user, find_user_by_id, find_user_by_username_password, update_user_interests, \
    find_all_articles, find_one_article
from main import get_recommendation

views = Blueprint(__name__, "views")
views.secret_key = 'utku123123'


@views.route("/")
def home():
    return render_template("login.html")


@views.route("/register")
def register():
    return render_template("register.html")


@views.route('/register-user', methods=['POST'])
def signup():
    new_username = request.form['new_username']
    new_password = request.form['new_password']
    preferences = request.form['preferences']

    add_user(new_username, new_password, preferences)
    return redirect('/')


@views.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    user = find_user_by_username_password(username, password)

    if user and not None:
        session['user_id'] = str(user['_id'])
        return render_template('index.html')
    else:
        error_message = "Kullanıcı Adı veya Şifreniz Hatalı."
        return render_template('login.html', error=error_message)


@views.route('/get_user_interests')
def get_user_interests():
    user_id = session.get('user_id')
    user = find_user_by_id(user_id)

    if user is None:
        return jsonify({'error': 'Kullanıcı bulunamadı'}), 404

    preferences = user['preferences']
    username = user['username']
    return jsonify({'username': username, 'preferences': preferences})


@views.route('/update_profile', methods=['POST'])
def update_profile():
    if request.method == 'POST':
        data = request.json

        username = data.get('username')
        new_interests = data.get('interests')

        user_id = session.get('user_id')
        update_user_interests(user_id, new_interests)

        return jsonify({'message': 'Profil başarıyla güncellendi'}), 200
    else:
        return jsonify({'error': 'Yanlış istek metodu'}), 405


@views.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    user_id = session.get('user_id')
    user = find_user_by_id(user_id)

    if user is None:
        return jsonify({'error': 'Kullanıcı bulunamadı'}), 404

    scibert_titles, fasttext_titles = get_recommendation(user, find_all_articles())

    return jsonify({'scibert_titles': scibert_titles, 'fasttext_titles': fasttext_titles})


@views.route('/filter', methods=['GET'])
def filter_article():
    keyword = request.args.get('keyword')
    article = find_one_article(keyword)

    if article:
        return jsonify({'article': {'Title': article['Title'], 'Text': article['Text']}})
    else:
        return jsonify({'error': 'No matching articles found.'})
