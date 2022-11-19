from flask import Flask
from flask import request
from flask import render_template
import pickle 
import numpy as np
with open("predict_population.pickle", mode="rb") as fp:
    model = pickle.load(fp)
model.predict(np.array([[1300, 4000]]))


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calc", methods=["POST"])
def calculation():
    num1 = int(request.form["num1"])
    num2 = int(request.form["num2"])
    result = int(model.predict(np.array([[num1, num2]])))
    return render_template("result.html", result=result, num1=num1, num2=num2)


app.run()