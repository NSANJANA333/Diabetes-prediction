from flask import Flask, request, render_template, flash
import pickle
import numpy as np
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret'

# scaler = pickle.load(open('scaledModel.pkl', "rb"))
# model = pickle.load(open('svmMmodel.pkl', "rb"))

with open("svmMmodel.pkl","rb") as f :
    model = pickle.load(f)

with open("scaledModel.pkl","rb") as f :
    std_scalar = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        gender = request.form.get('gender')
        if gender == "Male":
            gender = 1
        elif gender == "Female" :
            gender = 0

        age = int(request.form.get('age'))

        hyper = request.form.get('hyper')
        if hyper == "yes":
            hyper = 1
        elif hyper == "no" :
            hyper = 0

        heart = request.form.get('heart')
        if heart == "yes":
            heart = 1
        elif heart == "no":
            heart = 0

        smoke = request.form.get('smoke')
        if smoke == "non_smoker":
            smoke = 0
        elif smoke == "past_smoker":
            smoke = 1
        elif smoke == "current":
            smoke = 2

        bmi = float(request.form.get('bmi'))
        hlevel = float(request.form.get('hlevel'))
        glucose = int(request.form.get('glucose'))

        input_features = [[gender, age, hyper, heart, smoke, bmi, hlevel, glucose]]
        scaler_value = std_scalar.transform(input_features)
        prediction = model.predict(scaler_value)

    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
