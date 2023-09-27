import json
import string
import random
import nltk
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

print(tf.__version__)
print(pd.__version__)
print(np.__version__)
print(keras.__version__)
print(nltk.__version__)
print("end")
nltk.download("punkt")
# nltk.download("wordnet")
nltk.download('omw-1.4')
#nltk.download("popular")

def ourText(text):
    newTkns = nltk.word_tokenize(text)
    newTkns = [lm.lemmatize(word) for word in newTkns]
    return newTkns

def wordBag(text, vocab):
    newTkns = ourText(text)
    bagOfWords = [0] * len(vocab)
    for w in newTkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOfWords[idx] = 1
    return np.array(bagOfWords)

def Pclass(text, vocab, labels):
    bagOfWords = wordBag(text, vocab)
    ourResult = ourNewModel.predict(np.array([bagOfWords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]
    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList

def getRes(firstList, fJson):
    tag = firstList[0]
    listOfIntents = fJson["ourIntents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            ourResult = random.choice(i["responses"])
            break
    return ourResult

def remove_punc(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def is_yes(text):
    words = ["yes", "yeah", "y" , "yea", "yep", "sure", "certainly", "indeed", "of course", "absolutely"]
    return any(word in remove_punc(text.lower()) for word in words)

# Read the Excel file
data_frame = pd.read_excel("C:/Users/sandeep's laptop/Desktop/intents_data.xlsx")

# Create the dictionary
ourData = {"ourIntents": []}

# Iterate over the rows in the DataFrame
for index, row in data_frame.iterrows():
    tag = row['tag']
    pattern = row['patterns']
    response = row['responses']

    # Create a new intent dictionary
    intent = {
        "tag": tag,
        "patterns": pattern.split(',,'),
        "responses": response.split(',,')
    }

    # Add the intent to ourData
    ourData["ourIntents"].append(intent)

# Save the ourData dictionary as a JSON file
with open("intents.json", "w") as file:
    json.dump(ourData, file)

lm = WordNetLemmatizer()


# Lists
ourClasses = []
newWords = []
documentX = []
documentY = []

# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
for intent in ourData["ourIntents"]:
    for pattern in intent["patterns"]:
        ourNewTkns = nltk.word_tokenize(pattern)
        newWords.extend(ourNewTkns)
        documentX.append(pattern)
        documentY.append(intent["tag"])

    if intent["tag"] not in ourClasses:
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
newWords = sorted(set(newWords))
ourClasses = sorted(set(ourClasses))

trainingData = []
outEmpty = [0] * len(ourClasses)

# Bag-of-words model
for idx, doc in enumerate(documentX):
    bagOfWords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfWords.append(1) if word in text else bagOfWords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfWords, outputRow])

random.shuffle(trainingData)
trainingData = np.array(trainingData, dtype=object)

test = "What time is it"


x = np.array(list(trainingData[:, 0]))
y = np.array(list(trainingData[:, 1]))

iShape = (len(x[0]),)
oShape = len(y[0])

ourNewModel = Sequential()
ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
ourNewModel.add(Dropout(0.5))
ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation="softmax"))

md = tf.keras.optimizers.Adam(learning_rate=0.01)
ourNewModel.compile(loss='categorical_crossentropy',
                    optimizer=md,
                    metrics=["accuracy"])

ourNewModel.fit(x, y, epochs=27, verbose=1)


# Save the model weights
ourNewModel.save_weights("model_weights.h5")

# Save the vocabulary and classes
vocab_classes = {"vocab": newWords, "classes": ourClasses}
with open("vocab_classes.json", "w") as file:
    json.dump(vocab_classes, file)
