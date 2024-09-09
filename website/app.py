from flask import Flask
from views import views

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")
app.secret_key = 'utku123123'

if __name__ == '__main__':
    app.run(debug=True, port=8000)
