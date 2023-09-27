import tkinter as tk
from tkinter import ttk
import json
import os
# from pyplot import matplotlib
import numpy as np
import string
import nltk
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import logging
import sys
import re
print(sys.version)


logging.basicConfig(filename='chat.log', level=logging.DEBUG)

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
   print("Script is packaged as executable")
   app_root_dir = sys._MEIPASS
else:
   print("Script is running as a script")
   app_root_dir = '.'


nltk.download("punkt")
#nltk.download("popular")
nltk.download('omw-1.4')

#for logging the questions
def log_chat_message(message):
    logging.info(message)
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
# def getRes(firstList, fJson):
#     if not firstList:
#         return "Sorry, I didn't understand your question."

#     tag = firstList[0]
#     listOfIntents = fJson["ourIntents"]
#     for i in listOfIntents:
#         if i["tag"] == tag:
#             ourResult = random.choice(i["responses"])
#             break
#     else:
#         ourResult = "Sorry, I don't have an appropriate response for that."

#     return ourResult

def remove_punc(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# def is_yes(text):
#     words = ["yes", "yeah", "y" , "yea", "yep", "sure", "certainly", "indeed", "of course", "absolutely"]
#     return any(word in remove_punc(text.lower()) for word in words)

def answer(text):
    newMessage = str(text)
    intents = Pclass(newMessage, newWords, ourClasses)
    if intents == []:
        return [],None
    # print("Intents:", intents,newMessage)
    ourResult = getRes(intents, ourData)
    return intents, ourResult

def yes_click(response):
    result_text.configure(state='normal')
    print("blah",response)
    result_text.insert(tk.END, "{intent} \n".format(intent=response))
    result_text.configure(state='disabled')

def no_click(question):
    confirmation_text.configure(state='normal')
    confirmation_text.delete('1.0', tk.END)
    confirmation_text.insert(tk.END, "Please rephrase the question in a different way or enter the possible key phrases for the question\n I am storing the question which you can share so that I can become more efficient,")
    log_chat_message(question)
    confirmation_text.configure(state='disabled')

def wait_for_button_click(button1, button2):
    window = tk.Toplevel()
    window.geometry("1x1-0-0")

    var = tk.IntVar()
    button1.config(command=lambda: var.set(1))
    button2.config(command=lambda: var.set(2))
    window.wait_variable(var)

    window.destroy()
    return var

def myClick(button1, button2):
    query = text_box.get()
    text_box.delete(0, tk.END)
    erase_click()

    if query.replace(" ", "") == "":
        return
    else:
        result_text.configure(state='normal')
        map_tag, map_answer = answer(query)
        # print(map_answer)

        confirmation_text.configure(state='normal')
        if map_tag != [] and map_tag[0] != 'greeting' and map_tag[0] !='calculator':
            confirmation_text.insert(tk.END, "I see you have asked the question about {tag}? \n Is it correct? \n".format(tag=map_tag[0]))
            confirmation_text.configure(state='disabled')
            send_button.configure(state= 'disabled')
            button1.configure(state='normal')
            button2.configure(state='normal')
            response = wait_for_button_click(button1, button2)
            if response.get() == 1:
                yes_click(map_answer)
            else:
                no_click(query)

            button1.configure(state='disabled')
            button2.configure(state='disabled')
            result_text.configure(state='disabled')
            text_box.delete(0, tk.END)
        elif map_tag == ['calculator']:
            # Extract the mathematical expression using regular expression
            math_expr_match = re.search(r'(\d+(\.\d+)?)\s*([-+*/])\s*(\d+(\.\d+)?)', query)
        
            if math_expr_match:
                # Extract the numerical values and the operator
                num1 = float(math_expr_match.group(1))
                operator = math_expr_match.group(3)
                num2 = float(math_expr_match.group(4))
        
                # Evaluate the expression
                try:
                    if operator == '+':
                        result = num1 + num2
                    elif operator == '-':
                        result = num1 - num2
                    elif operator == '*':
                        result = num1 * num2
                    elif operator == '/':
                        result = num1 / num2
                    else:
                        raise ValueError("Invalid operator")
        
                    confirmation_text.insert(tk.END, "Result : {}\n for query : {}".format(result,query))
                except ValueError as e:
                    confirmation_text.insert(tk.END, "Invalid operator or calculation.\n")
            else:
                confirmation_text.insert(tk.END, "Please provide a valid calculation.\n")

       
            
        elif map_tag == ['greeting']:
            confirmation_text.insert(tk.END, map_answer)
            confirmation_text.configure(state='disabled')
            
        else:
            confirmation_text.insert(tk.END, "It seems I dont have an answer to your question yet. I am sending your question to the development team. Thank You.")
            log_chat_message(query)
            confirmation_text.configure(state='disabled')
        send_button.configure(state= 'normal')
        
        
        

        

def erase_click():
    confirmation_text.configure(state='normal')
    result_text.configure(state='normal')
    confirmation_text.delete('1.0', tk.END)
    result_text.delete('1.0', tk.END)
    confirmation_text.configure(state='disabled')
    result_text.configure(state='disabled')

def on_entry_click(event):
    if text_box.get() == "Enter your query":
        text_box.delete(0, tk.END)
        text_box.insert(0, "")

# Load the vocabulary and classes
with open(os.path.join(app_root_dir, "vocab_classes.json"), "r") as file:
    vocab_classes = json.load(file)

newWords = vocab_classes["vocab"]
ourClasses = vocab_classes["classes"]

lm = WordNetLemmatizer()

ourNewModel = Sequential()
ourNewModel.add(Dense(128, input_shape=(len(newWords),), activation="relu"))
ourNewModel.add(Dropout(0.5))
ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(len(ourClasses), activation="softmax"))

ourNewModel.load_weights(os.path.join(app_root_dir, "model_weights.h5"))

# Load the ourData dictionary
with open(os.path.join(app_root_dir, "intents.json"), "r") as file:
    ourData = json.load(file)

root = tk.Tk()
root.title("Query bot for NGT")
#root.state('zoomed')
root.geometry("1000x750")
root.configure(bg='black')




style = ttk.Style()
style.theme_use("classic")

text_box = ttk.Entry(root, width=100)
text_box.insert(0, "Enter your query")
text_box.grid(row=0, columnspan=3)
text_box.configure(foreground="gray")

send_button = ttk.Button(root, text="Send", width=10)
send_button.grid(row=0, column=3)

button1 = ttk.Button(root, text="Yes", width=10, state='disabled')
button2 = ttk.Button(root, text="No", width=10, state='disabled')
button3 = ttk.Button(root, text="Clear", width=10)
button1.grid(row=1, column=0)
button2.grid(row=1, column=1)
button3.grid(row=1, column=2)

button3.config(command=erase_click)

confirmation_text = tk.Text(root, height=4, width=100)
#confirmation_text.insert(tk.END, "Welcome! Please enter your query.\n")
confirmation_text.configure(state='disabled')
confirmation_text.grid(row=2, columnspan=4, padx=10, pady=10)

result_text = tk.Text(root, height=30, width=100)
result_text.configure(state='disabled')
result_text.grid(row=3, columnspan=4, padx=10, pady=10)

button1.configure(command=lambda: yes_click(result_text.get("1.0", "end-1c")))
button2.configure(command=no_click)
button3.configure(command=erase_click)
send_button.configure(command=lambda: myClick(button1, button2))
text_box.bind('<FocusIn>', on_entry_click)

root.mainloop()