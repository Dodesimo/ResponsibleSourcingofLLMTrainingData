'''
What does this script do:

- Roberta Model encodes two knowledge graph relationships (one from the source document, another from the LLM continuation)
- Encoding stored in a Pinecone database.
- Semantic simliarity calculated between each group of encodings,  adhering to the thresholds.


'''
from pinecone import Pinecone
# importing the required libraries
from sentence_transformers import SentenceTransformer
import torch
import csv
from datetime import datetime
from time import time

# opening the file with the knowledge represented by sentences
#   and loading it into a list
KG_sentences = []

with open('knowledge_sentences.csv', newline='') as inputfile:
    csv_reader = csv.reader(inputfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        KG_sentences.append(row[0])


# setting the threshold for the single match semantic proximity score
min_single_sem_prox_score = 0.6

# setting the % of phrases to be matched for the query to be considered compatible with the knowledge base
perc_min_single_sem_prox_score = 0.1

# determining the number of sentences to be matched for the query to be considered
#   compatible with the knowledge base
num_KB_sentences = round(len(KG_sentences) * perc_min_single_sem_prox_score,2)

# calculating the actual threshold for the query to be considered
#   compatible with the knowledge base
min_cumulative_sem_prox_score = min_single_sem_prox_score * num_KB_sentences
print ('min_cumulative_sem_prox_score:', min_cumulative_sem_prox_score)

# defining the continuation sentences
C_sentences = []
with open('knowledge_cont_sentences.csv', newline='') as inputfile:
    for row in csv.reader(inputfile):
        C_sentences.append(row[0])


# === preparing pinecone and loading the model to create the embeddings
# initializing pinecone
pc = Pinecone(api_key="86d58470-386b-415a-a29b-9196f0615a93",
              environment="us-east-1")
#active_indexes = pc.list_indexes()
#index_description = pc.describe_index(active_indexes[0])
index_name = 'myindex'

# loading the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-roberta-large-v1'
model = SentenceTransformer(model_name)

# creating the Pinecone index
index = pc.Index(index_name)

# === embedding the text
# printing starting time
start_time = datetime.now()
print ('\n-Starting the embedding process for the knowledge base at {}'.format(start_time), '\n')

t = time()

# creating the embeddings for the input text
corpus_embedding = model.encode(KG_sentences).tolist()
print('Time to embed the input text: {} mins'.format(round((time() - t) / 60, 4)),'\n')
#print ('len of the corpus embeddings: ', len(corpus_embedding))

# creating the embeddings for the query sentences
cont_embeddings = model.encode(C_sentences).tolist()


# === adding the knowledge base text embeddings to the pinecone index
#   IDs for the vectors would be added. They are 1 to n

# creating a list of (id, vector) tuples
#    generating unique IDs for each vector
ids = [str(i) for i in range(len(corpus_embedding))]
#metadata_label = ['sentence:']
data = list(zip(ids, corpus_embedding))
#index_description = pc.describe_index("myindex")
#print (index_description)
# upserting the corpus embeddings to the index
index.upsert(data)

t = time()

# === querying the database/index with the query sentences

# looping over the query embeddings and sentences
for query_embedding, query_sentence in zip(cont_embeddings, C_sentences):
    print("\n---Query:", query_sentence)
    count_matches = 0
    res = index.query(vector= query_embedding, top_k=3, include_values=True)
    # printing the matched sentences and their scores
    cumulative_score = 0
    for res in res.matches:
        if res.score < min_single_sem_prox_score:
            continue
        print ("   Matched sentence:", KG_sentences[int(res.id)])
        print("    - score:", round(res.score,2))
        cumulative_score = cumulative_score + res.score
        count_matches =+ 1
    if cumulative_score == 0:
        print ("  -- No match: this phrase in the continuation is not compatible with the knowledge base.")
    else:
        print ("   ---> The semantic proximity of this phrase in the continuation to the knowledge base is:", round(cumulative_score,2))
        if cumulative_score < min_cumulative_sem_prox_score:
            print ('        That means the phrase is not compatible with the knowledge base')
            if count_matches > 0:
                print('        but there is some minimal compatibility')
        else:
            print('        That means the phrase is compatible with the knowledge base')

print('\nTime to evaluate the matching: {} mins'.format(round((time() - t) / 60, 4)))
print(cumulative_score)


#NLTK --> really old, find a new way to figure out how to figure out how many sentences there are.
#Lovane Community detection --> recreate sentences at the graph level (comparison between source and continuation information).
#Do this week
#PostGres (langchain), as well as play around with metric


