from keras.engine.saving import model_from_json
from sklearn.feature_extraction.text import CountVectorizer
import logging, os, yaml
import numpy as np
from googletrans import Translator

logging.getLogger('tensorflow').disabled = True
translator = Translator()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

dir_path = 'chatterbotenglish'
files_list = os.listdir(dir_path + os.sep)

questions = list()
answers = list()
for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 2:
            questions.append(con[0])
            replies = con[1:]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append(ans)
        elif len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])

# num 558
vectorizer = CountVectorizer()
vectorizer.fit(questions)

y_dict = dict(zip(range(len(answers)), answers))

while True:
    question = input("Pregunta: ")

    if question == 'exit':
        break

    lang_info = translator.translate(question, dest="en")
    if lang_info.src != 'en':
        question = lang_info.text

    question_vec = vectorizer.transform([question])
    prediction = loaded_model.predict(question_vec[:1])
    idx = np.argmax(prediction[0])

    if lang_info.src != 'en':
        print("Respuesta:", translator.translate(y_dict[idx], dest=lang_info.src).text)
    else:
        print("Respuesta:", y_dict[idx])
