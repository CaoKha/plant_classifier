from flask import Flask, render_template, request, url_for, jsonify
from fastai.vision.all import *
import re
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

learn = load_learner('./models/model.pkl')

@app.route('/')
def homepage():
    return render_template('index.html')



def predict_single(img_file):
    '''function to take image and return prediction'''
    pred, pred_idx, probs = learn.predict(PILImage.create(img_file))
    probs_list = [pred, probs[pred_idx].tolist()]
    return probs_list


@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        my_prediction = predict_single(request.files['inputImage'])
        plant_class = str(my_prediction[0])
        plant_type = re.search(r'.+?(?=_)',plant_class).group()
        growth_stage = re.search(r'(?<=_)[\w+.-]+',plant_class).group()
        probability = str("{:.2f}".format(float(my_prediction[1])*100))
    return render_template('results.html', prediction = {
        "plant_type":plant_type,
        "growth_stage": growth_stage,
        "probability": probability,
    }, comment='asd')


if __name__ == '__main__':
    app.run(debug=True)
