from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForQuestionAnswering
from datasets import load_dataset

with open('questions.txt', 'r') as f:
    questions = f.read()

with open('context.txt', 'r') as f:
    context = f.read()

questions = questions.split('\n')[:-1]

models = ('deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad', 'twmkn9/bert-base-uncased-squad2')

for model_name in models:
    print('with model: ',model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    for q in questions:
        print(q)
        ans = nlp({
            'question': q,
            'context': context
        })
        print(ans['answer'])
        print()