from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset


modelname = 'distilbert-base-uncased-distilled-squad'

model = AutoModelForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
squad = load_dataset('squad', split='validation')

def find_context(question, source):
    scores, retrieved_examples = source.get_nearest_examples("context", question, k=1)
    try:
        context = retrieved_examples['context'][0]
    except:
        context = None
    # print('context', context)
    return context

gpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

while True:
    question = input(">>> ")
    if question.lower() == 'bye':
        print('Bye and have a good day!')
        break
    while not question:
        print('Prompt should not be empty!')
        question = input(">>> ")
    
    context = find_context(question, squad)

    if context is None:
        user_inputs = gpt_tokenizer.encode(question + gpt_tokenizer.eos_token,\
                                                return_tensors='pt')
        gpt_predict = gpt_model.generate(user_inputs, max_length=1000,pad_token_id=gpt_tokenizer.eos_token_id)
        anw = gpt_tokenizer.decode(gpt_predict[:, user_inputs.shape[-1]:][0], skip_special_tokens=True)
        print(anw)

    else:
        anw = nlp({
            'question': question,
            'context': context
        })
        print(anw['answer'])


