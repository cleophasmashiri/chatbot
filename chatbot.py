from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = 'facebook/blenderbot-400M-distill'

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

conversation_hist = []

while True:
    history_str =  "\n".join(conversation_hist)

    input_text = input('> ')

    inputs = tokenizer.encode_plus(history_str, input_text, return_tensors='pt')

    outputs = model.generate(**inputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_hist.append(input_text)
    conversation_hist.append(response)

    print(response)
