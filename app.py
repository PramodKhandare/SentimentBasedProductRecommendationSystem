from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import *


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():

    user = request.form['userName']

    user = user.lower()
    items = get_Recommendations(user)
    
    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)

        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message="We can not recommend for this user. Please try for the suggested users.")

if __name__ == '__main__':
    app.run()