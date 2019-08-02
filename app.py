from flask import Flask,render_template,url_for,request
from keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import tensorflow as tf

app = Flask(__name__)


def init():
    global clf, graph
    clf = load_model('spam_model.h5')
    graph = tf.get_default_graph()

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#Alternative Usage of Saved Model
	with open('tokenizer.json') as f:
		data = json.load(f)
		tokenizer = tokenizer_from_json(data)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = tokenizer.texts_to_sequences(data)
		vect = pad_sequences(vect, padding='post', maxlen=100)
		with graph.as_default():
			prediction = clf.predict_classes(vect)
		if prediction == [[0]]:
			my_prediction = 0
		else:
			my_prediction = 1
	return render_template('result.html', prediction=my_prediction)



if __name__ == '__main__':
	init()
	app.run(debug=True)