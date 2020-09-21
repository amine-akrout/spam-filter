import pandas as pd
import numpy as np
#read data
df = pd.read_csv('data/emails.csv', encoding='latin-1')
df.head()

# Cleaning the texts
''''
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Every mail starts with 'Subject :' will remove this from each text
sw = set(stopwords.words('english'))
df['text']= df['text'].map(lambda text: text[7:])
df['text'] = df['text'].map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ', text)).apply(lambda x: (x.lower()))

def filter_stop_words(train_sentences):
    train_sentences = ' '.join(word for word in train_sentences.split() if word not in sw)
    return train_sentences

df['text'] = df['text'].apply(filter_stop_words)

'''

#Split data train/test
from sklearn.model_selection import train_test_split
y=df['spam']
sentences_train, sentences_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.3, random_state=1234)
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index) + 1

from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Calculate the weights for each class so that we can balance the data
from sklearn.utils import class_weight
#weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

## Fit the model
model.fit(X_train, y_train, validation_split=0.2, epochs=5) #, class_weight=weights

###Test model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
matrix = confusion_matrix(y_test, y_pred)


#save tokenizer
import json
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
#save model
model.save('spam_model.h5')  # creates a HDF5 file 'my_model.h5'

del model  # deletes the existing model


clf = load_model('spam_model.h5')
comment = "this is a test"
data = [comment]
vect = tokenizer.texts_to_sequences(data)
vect = pad_sequences(vect, padding='post', maxlen=100)
prediction = clf.predict_classes(vect)
