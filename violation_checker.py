import json
import random
import numpy as np

import nltk
import tflearn
import tensorflow as tf

stemmer = nltk.LancasterStemmer()
data = None

with open('data1.json') as json_data:
    data = json.load(json_data)
    print(data)
categories = list(data.keys())
words = []

docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
       w = nltk.word_tokenize(each_sentence)
    print ("tokenized words: ", w)
    words.extend(w)
    docs.append((w, each_category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print (words)
print (docs)

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)

for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print(train_x)
print("/n")
print(train_y)

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=10000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# let's test the mdodel for a few sentences:
sent_1 = "The Bodubala Sena Organization starts the Anti-muslim violence in sri lanka and done a protest this monday."
sent_2 = "A police officer attached to Thelippalai Police Station has committed suicide using his service weapon."



def get_tf_record(sentence):
    global words


    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return (np.array(bow))
# we can start to predict the results for each of the 4 sentences
print(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_2)]))])


