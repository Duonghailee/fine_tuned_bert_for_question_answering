# fine_tuned_bert_for_question_answering
This is the repo for fine tuning a bert based model for downstream task: question answering.
The whole notebook codes are referenced from Hugginface published notebook on [colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb#scrollTo=-t1DxGvTuGFp) , with slightly modifications for research's experiment purpose only. We keep the valuation part form the original notebook untoched and use it to evaluate our fine-tuned models. Other parts have been slightly modified.Thanks Huggingface team.

In order to run gpt_and_squad_QA.py which is an attempt to open domain chatbot interactive mode, Elastic search has to be installed, preferably running the Elastic search with docker. 

Using docker:
First start your docker app, then run
`docker network create elastic`

`docker pull docker.elastic.co/elasticsearch/elasticsearch:7.12.1`

`docker run --name es01-test --net elastic -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.12.1`


