from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

app = Flask(__name__)

def init():
    global clf
    clf = load_model('spam_model.h5')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	with open('tokenizer.json') as f:
		data = json.load(f)
		tokenizer = tokenizer_from_json(data)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = tokenizer.texts_to_sequences(data)
		vect = pad_sequences(vect, padding='post', maxlen=100)
		prediction = clf.predict_classes(vect)
		if prediction == [[0]]:
			my_prediction = 0
		else:
			my_prediction = 1
	return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	init()
	app.run(debug=True,host='0.0.0.0')