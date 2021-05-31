import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Lambda, Bidirectional, Dense, Dropout, LSTM

path = 'dataset\mpst_full_data.csv'
preprocessed_path = 'dataset\dataset.csv'
lemmatizer = WordNetLemmatizer()

def import_dataset(path):
    return pd.read_csv(path)

def max_len(x):
    a=x.split()
    return len(a)

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def remove_spaces(x):
    x=x.split(",")
    nospace=[]
    for item in x:
        item=item.lstrip()
        nospace.append(item)
    return (",").join(nospace)

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

try:
    df = import_dataset(preprocessed_path)
except:
    print("No preprocessed dataset.")

    df = import_dataset(path)
    pd.set_option('max_columns', None)

    preprocessed_synopsis = []

    for sentence in df['plot_synopsis'].values:
        sentence = decontracted(sentence)
        sentence = re.sub("\S*\d\S*", "", sentence).strip()
        sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
        sentence = lemmatize_sentence(sentence)
        preprocessed_synopsis.append(sentence.strip())

    df['preprocessed_plots']=preprocessed_synopsis
    df.to_csv('dataset\\dataset.csv')

df['tags'] = df['tags'].apply(remove_spaces)

train_df=df.loc[df.split=='train']
train_df=train_df.reset_index()
test_df=df.loc[df.split=='test']
test_df=test_df.reset_index()
val_df=df.loc[df.split=="val"]
val_df=val_df.reset_index()

results = set()
train_df['tags'].str.lower().str.split(",").apply(results.update)
categories_num = len(results)
print("Number of categories: ", categories_num)

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(","), binary='true')
y_train = vectorizer.fit_transform(train_df['tags']).toarray()
y_test = vectorizer.transform(test_df['tags']).toarray()

# print(max(df['plot_synopsis'].apply(max_len)))

vect=Tokenizer()
vect.fit_on_texts(train_df['preprocessed_plots'])
vocab_size = len(vect.word_index) + 1
# print(vocab_size)

encoded_docs_train = vect.texts_to_sequences(train_df['preprocessed_plots'])
max_length = vocab_size
padded_docs_train = pad_sequences(encoded_docs_train, maxlen=1200, padding='post')
# print(padded_docs_train)

encoded_docs_test = vect.texts_to_sequences(test_df['preprocessed_plots'])
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=1200, padding='post')
encoded_docs_cv = vect.texts_to_sequences(val_df['preprocessed_plots'])
padded_docs_cv = pad_sequences(encoded_docs_cv, maxlen=1200, padding='post')

"""text_clf_svm = Pipeline([('vect', CountVectorizer(min_df=0, lowercase=True, stop_words=stopwords, max_df=0.80)),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', LinearSVC())])

_ = text_clf_svm.fit(train_df['plot_synopsis'].values, train_df['tags'].values)
predicted_svm = text_clf_svm.predict(test_df['plot_synopsis'].values)
print("Accuracy with LinearSVC: " + str(numpy.mean(predicted_svm == test_df['tags'].values)))

text_clf_nb = Pipeline([('vect', CountVectorizer(min_df=0, lowercase=True, stop_words=stopwords, max_df=0.80)),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', MultinomialNB())])

_ = text_clf_nb.fit(train_df['plot_synopsis'].values, train_df['tags'].values)
predicted_nb = text_clf_nb.predict(test_df['plot_synopsis'].values)
print("Accuracy with MultinomialNB: " + str(numpy.mean(predicted_nb == test_df['tags'].values)))
"""

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

def ELMoEmbedding(input_text):
    return elmo(tf.reshape(tf.cast(input_text, tf.string), [-1]), signature="default", as_dict=True)["elmo"]

def create_model():
    """input_layer = Input(shape=(1,), dtype="string")
    embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024,))(input_layer)
    BiLSTM = Bidirectional(layers.LSTM(1024, return_sequences=False, recurrent_dropout=0.2, dropout=0.2))(embedding_layer)
    Dense_layer_1 = layers.Dense(8336, activation='relu')(BiLSTM)
    Dropout_layer_1 = layers.Dropout(0.5)(Dense_layer_1)
    Dense_layer_2 = layers.Dense(4168, activation='relu')(Dropout_layer_1)
    Dropout_layer_2 = layers.Dropout(0.5)(Dense_layer_2)
    output_layer = layers.Dense(1, activation='sigmoid')(Dropout_layer_2)
    model = Model(inputs=[input_layer], outputs=output_layer)
    """
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    #BiLSTM = Bidirectional(LSTM(1024, return_sequences=False, recurrent_dropout=0.2, dropout=0.2))(embedding)
    #dense1 = Dense(8336, activation='relu')(BiLSTM)
    #dropout = Dropout(0.5)(dense1)
    dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
    pred = Dense(categories_num, activation='sigmoid')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    """
    # create the model
    embedding_vector_length = 64
    model = keras.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_vector_length, input_length=1200))
    # TODO: Maybe try LeakyRELU activation on Conv1D instead of relu
    # TODO: more Conv1Ds
    model.add(Conv1D(filters=embedding_vector_length, kernel_size=5, padding='same', activation=layers.LeakyReLU(alpha=0.1)))
    model.add(layers.MaxPooling1D(pool_size=2, strides=1))
    # define LSTM model
    # TODO: Maybe add tanh activation on one of LSTM layers
    model.add(Bidirectional(layers.LSTM(1024, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    # Adding a dropout layer
    # TODO: Less dropout and LSTM with more inputs
    # TODO: Bidirectional LSTM
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.1))
    # TODO: Maybe one more Dense layer with more neurons (often when you don't have enough neurons in the layer before the output layer, the model loses accuracy)
    # Adding a dense output layer with sigmoid activation
    model.add(layers.Dense(categories_num, activation='sigmoid'))"""

    print(model.summary())

    METRICS = [

        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    return model

def train(model):
    history = model.fit(padded_docs_train, y_train,
                        epochs=10,
                        verbose=1,
                        validation_data=(padded_docs_test, y_test),
                        batch_size=16)
    #TODO: try with different batch sizes

    model.save('model')

try:
    reconstructed_model = keras.models.load_model("model")
    model = reconstructed_model
    print("Loaded model.")
except:
    print("No model saved. Training a new one.")
    model = create_model()
    X = np.array(train_df['preprocessed_plots'])
    #train(model)
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = model.fit(X, y_train, epochs=5, batch_size=256, validation_split = 0.2)
        model.save_weights('./model_elmo_weights.h5')


predictions = model.predict([padded_docs_test])
thresholds = [0.1, 0.2, 0.3, 0.4]# ,0.5, 0.6, 0.7, 0.8, 0.9]

for val in thresholds:
    print("For threshold: ", val)
    pred = predictions.copy()

    pred[pred >= val] = 1
    pred[pred < val] = 0

    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')

    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

def predict_sample():
    t = val_df.sample(1)
    encoded_docs = vect.texts_to_sequences(t['preprocessed_plots'])
    padded_docs = pad_sequences(encoded_docs, maxlen=1200, padding='post')
    pred = model.predict(padded_docs).tolist()

    for i in range(len(vectorizer.inverse_transform(pred[0])[0])):
        print(pred[0][i], "-->", vectorizer.inverse_transform(pred[0])[0][i])

    for i in range(len(pred[0])):
        if (pred[0][i] < 0.1):
            pred[0][i] = 0
        else:
            pred[0][i] = 1

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    print("Movie title -->", t['title'].values)
    print("Synopsis -->", t['plot_synopsis'].values)
    print("Original tags -->", t['tags'].values)
    print("Predicted tags -->", vectorizer.inverse_transform(pred[0])[0])


predict_sample()
predict_sample()
predict_sample()
