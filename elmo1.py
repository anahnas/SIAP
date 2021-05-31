import re
import string

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn import preprocessing, model_selection
import keras
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
import classification
import sys
import spacy
from spacy.lang.en import English
import matplotlib.pyplot as plt
from wordcloud import WordCloud

spacy.load('en_core_web_sm')
parser = English()

# np.set_printoptions(threshold=sys.maxsize)

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

path = 'dataset\mpst_full_data.csv'
preprocessed_path = 'dataset\dataset.csv'

# Stop words and special characters
STOPLIST = set(stopwords.words('english'))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "''"]


def plot_word_cloud(text):
    wordcloud_instance = WordCloud(width=800, height=800,
                                   background_color='black',
                                   stopwords=STOPLIST,
                                   min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_instance)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Data Cleaner and tokenizer
def tokenizeText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()

    tokens = parser(text)

    # lemmatization
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # reomve stop words and special charaters
    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    tokens = [tok for tok in tokens if len(tok) >= 3]

    # remove remaining tokens that are not alphabetic
    tokens = [tok for tok in tokens if tok.isalpha()]

    tokens = list(set(tokens))

    return ' '.join(tokens[:])

try:
    df = classification.import_dataset(preprocessed_path)
except:
    print("No preprocessed dataset.")

    df = classification.import_dataset(path)
    pd.set_option('max_columns', None)

    preprocessed_synopsis = []

    for sentence in df['plot_synopsis'].values:
        sentence = classification.decontracted(sentence)
        sentence = re.sub("\S*\d\S*", "", sentence).strip()
        sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in classification.stopwords)
        sentence = classification.lemmatize_sentence(sentence)
        preprocessed_synopsis.append(sentence.strip())

    df['preprocessed_plots'] = preprocessed_synopsis
    df.to_csv('dataset\\dataset.csv')

texts = ''
for index, item in df.iterrows():
    texts = texts + ' ' + item['plot_synopsis']

# plot_word_cloud(texts)

print(df['tags'].value_counts())
df['tags'] = df['tags'].apply(classification.remove_spaces)

labels = set()
df['tags'].str.lower().str.split(",").apply(labels.update)
categories_num = len(labels)
print("Number of categories: ", categories_num)

train_df = df.loc[df.split == 'train']
"""or df.split == "val"]"""
train_df = train_df.reset_index()
test_df = df.loc[df.split == 'test']
test_df = test_df.reset_index()

# Create datasets (Only take up to 150 words)
train_text = train_df['preprocessed_plots'].tolist()
train_text = [' '.join(t.split()[0:150]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_labels = train_df['tags'].tolist()

test_text = test_df['preprocessed_plots'].tolist()
test_text = [' '.join(t.split()[0:150]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['tags'].tolist()

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","), binary='true')
train_labels = vectorizer.fit_transform(train_df['tags']).toarray()
test_labels = vectorizer.transform(test_df['tags']).toarray()


n = round(len(train_text)/2)
train_text = train_text[:n]
train_labels = train_labels[:n]

test_text = test_text[0:50]
test_labels = test_labels[0:50]

def ELMoEmbbeding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embbeding = Lambda(ELMoEmbbeding, output_shape=(1024,))(input_text)
dense = Dense(256, activation='relu')(embbeding)
pred = Dense(categories_num, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model.fit(train_text, train_labels, epochs=500, batch_size=32)
    model.save_weights('./elmo-model.h5')

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model.h5')
    predicts = model.predict(test_text, batch_size=2)


print(predicts)
i = 0
for pred in predicts:
    max_index_col = np.argmax(pred, axis=0)
    print(test_text[i])
    true_labels = []

    labels_indexes = np.where(test_labels[i] == 1)
    for l in labels_indexes[0]:
        true_labels.append(list(labels)[l])

    print("Real labels: ", true_labels)
    print("********Prediction*********: ", list(labels)[max_index_col], " (", np.max(pred),")")

    i = i + 1
