'''
This script takes 2 txt files, whose names are in a configuration file. They contain
    - the "assistant" prompt
    - the "user" prompt. This prompt represents the textual knowledge base

Using OpenAI API, extract a string (then transformed in a list) with triples of relationships like:
    [subject, predicate, object]

From the list, it creates a directed graph, that is a knowledge graph for the knowledge base.
The graph is saved as html via pyvis and as a gml
'''

# importing the required libraries
import pandas as pd
from openai import OpenAI
import networkx as nx
from pyvis.network import Network
import json
import spacy
import csv



# this function extracts relationships from text using OpenAI API
def extract_relationships(text1, text2):
    response = client.chat.completions.create(
        model="gpt-4",
        temperature = 1,
        top_p = 1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "user", "content": text2},
            {"role": "assistant", "content": text1}])
    #print ('whole response: \n', response)
    #print('relationships:', response.choices[0].message.content.strip(), '--\n')

    return response.choices[0].message.content.strip()

# this function lemmatize a list of words.
#   It is taking the list to lemmatize and the dataframe with the edges.
#   The edges will be updated with the new lemmatized strings
def lemmatize (list_to_lemmatize, df):
    nlp = spacy.load("en_core_web_lg")
    lemmata = []
    for token in list_to_lemmatize:
        elem = nlp(token)
        lemmata.append(elem.text)
        if elem.text != token:
            try:
                df.replace(to_replace=token, value=elem.text, inplace=True)
            except:
                continue

    return lemmata, df


# this function prepares the data for the knowledge graph creation from a list of lists whose elements are relationships
#   The dataframe has a format like: [subject, predicate, object]
#   It returns entities, relationships and the edge_list for the graph
def graph_prep (df):
    # creating a list of unique relationships
    relationships = df['predicate'].unique().tolist()
    # creating a list of unique entities, from the combination of "subjects" and "objects"
    entities = pd.concat([df['subject'], df['object']]).unique().tolist()

    # creating a list of edges
    edge_list = [] # initializing the list
    # iterating through the rows
    for index, row in df.iterrows():
        # converting each row to a list and add it to the list of rows
        row_as_list = list(row)
        edge_list.append(row_as_list)

    return entities, relationships, edge_list


# this function creates the knowledge graph from a list of lists whose elements are relationships
#   in a format like: [object, subject, predicate]
def graph_generation (edge_list):
    # creating the direct graph
    G = nx.DiGraph()
    for edge in edge_list:
        G.add_edge(edge[0], edge[2], label=edge[1])
    nx.draw(G, with_labels = True)

    # creating the html file to visualize the graph
    G_viz = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    G_viz.from_nx(G)
    G_viz.show_buttons(filter_=['physics'])
    graph_name = 'k_graph'
    G_viz.show(graph_name + '.html', notebook=False)

    # saving the graph as a gml file
    graph_file = graph_name + '.gml'
    nx.write_gml(G, graph_file)


### MAIN PROGRAM

# reading the configuration file
df = pd.read_csv('config.csv', header = None)

# assigning variables the values from the configuration file
api_key = df.iloc[0][1] # this is the OpenAI API key

in_assistant_text_file = df.iloc[1][1] # this is the name of the "assistant" text tile
in_user_text_file = df.iloc[2][1] # this is the name of the "user" text file, with the "knowledge base"
f = open(in_user_text_file)
firstLine = f.readline() # Get first line.
wordCount = 0
data = f.read()
lines = data.split()
wordCount += len(lines) #get word count
wordCount = wordCount - len(firstLine)

#Create OpenAI Client
client = OpenAI(api_key=api_key)

#Get autocomplete response.
response = client.chat.completions.create(
        model="gpt-4",
        temperature = 1,
        top_p = 1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "user", "content": f"Based off of your training, generate a continuation for the following sentence {firstLine}. The continuation must be {wordCount} long"}])

response.choices[0].message.content.strip()

with open("txt_cont.txt", "w") as file:
    file.write(response.choices[0].message.content.strip()
)