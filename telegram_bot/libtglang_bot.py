import telebot

import os.path
import sys
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

import onnxruntime as onnxrt
import numpy as np
from common import *
from code_snippet_model import *
from word_embeddings import *

MIN_PROB_THRESHOLD = 20 # Prob in %

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Укажите здесь токен вашего бота Telegram
TOKEN = '6582263948:AAHw_301JldCs6-qaBltHUAXEEfv2tQVykY'

# Создание объекта бота
bot = telebot.TeleBot(TOKEN)
fill_embeddings_map(ALPHABET)

# Обработка команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    print(message.chat.username)
    try:
        bot.send_message(message.chat.id, "Hello!\nI can understand what programming language you are speaking🤓")
    except:
        print('something went wrong')

@bot.message_handler(content_types=['text'])
def classify_input_text(message):
    onnx_session= onnxrt.InferenceSession("../code_snippet_clf.onnx")
    embedded_sentence = get_embedded_sentence(message.text)
    diff = TELEGRAM_MESSAGE_MAX_LEN - embedded_sentence.shape[1]
    embedded_sentence = np.append(embedded_sentence, np.zeros((ALPHABET_SIZE, diff)), axis=1)
    embedded_sentence = embedded_sentence.astype(np.float32)
    input_text = np.expand_dims(embedded_sentence, axis = 0)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_text})
    output = torch.tensor(result)
    label = targets_desctiption[torch.argmax(output)] # Post processing
    prob = round((output[0, 0, torch.argmax(output).item()] * 100).item())
    if (prob < MIN_PROB_THRESHOLD):
        bot.send_message(message.chat.id, "there is no code in the message you sent😤👺")
    else:
        text = f"Okay! I think this is {label} with probability {prob}%"
        bot.send_message(message.chat.id, text)

def start_bot():
    try:
        # Запуск бота
        bot.polling()
    except Exception as e:
        print(e)
        start_bot()

start_bot()
