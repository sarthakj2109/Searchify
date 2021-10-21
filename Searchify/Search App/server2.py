from flask import Flask, render_template, request
# from flask_cors import CORS
from predict2 import *
# from flask_ngrok import run_with_ngrok

import os
import threading

from flask import Flask
from pyngrok import ngrok
from flask import g
os.environ["FLASK_ENV"] = "development"

app = Flask(__name__)

#Create object of QnA class in predict1.py
obj=QnA()

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5001).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5001))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# Define Flask routes
@app.route('/')
def home():
  return render_template('homepage.html',url=public_url) # render default webpage

@app.route('/search', methods=['GET', 'POST'])
def search():
  errors = []
  results = {}
  pdftext=[]
  offset=[]
  if request.method == "POST":
    try:
      question = request.form.get('search')
      print(question)
      results,pdftext,offset = obj.predict(question)
      print(results)
      
    except Exception as e:
      print(e)
      errors.append(
        "Unable to get URL. Please make sure it's valid and try again."
        )
  return render_template('main_search.html', results=results, pdftext=pdftext,offset=offset,ques=question,url=public_url)

if __name__ == "__main__":
  app.run(port=5001,use_reloader=False)


