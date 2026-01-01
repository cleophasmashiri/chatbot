from flask import request, Flask, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
import json


MODEL_NAME = 'facebook/blenderbot-400M-distill'

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

conversation_hist = []

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/chatbot', methods=['POST'])
def chat():
    history_str =  "\n".join(conversation_hist)

    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data.get('prompt')

    inputs = tokenizer.encode_plus(history_str, input_text, return_tensors='pt')

    outputs = model.generate(**inputs, max_length=60)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_hist.append(input_text)
    conversation_hist.append(response)

    return response


if __name__=='__main__':
    app.run(host='0.0.0.0', port=50)
