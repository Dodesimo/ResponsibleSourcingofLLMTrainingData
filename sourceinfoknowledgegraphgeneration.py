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

# reading the files for "assistant" and "user"
with open(in_assistant_text_file, "r") as kb_text_assistant:
    text_assistant = kb_text_assistant.read()
with open(in_user_text_file, "r") as kb_text_user:
    text_user = kb_text_user.read()

# initializing the OpenAI API client
client = OpenAI(api_key=api_key)

# extracting relationships via OpenAI API
extracted_relationships_str = extract_relationships(text_assistant, text_user)

# converting the string from the API to a list of lists
extracted_relationships = json.loads(extracted_relationships_str)

# creating a pandas dataframe with ['subject', 'predicate', 'object'] as columns
columns_names = ['subject', 'predicate', 'object']
df1 = pd.DataFrame(extracted_relationships, columns = columns_names)

# lowering all the elements in the dataframe
for col_name in columns_names:
    df1[col_name] = df1[col_name].str.lower()

# calling the function to create the knowledge graph
entities, relationships, edge_list = graph_prep (df1)

#print ('\nentities:\n', entities)
#print ('\nrelationships:\n', relationships)
#print ('\nedge_list:\n', edge_list)


# placing the list of edges in a dataframe
#   This is done to have an easier way to replace values
df = pd.DataFrame(edge_list)

# calling the lemmatization function for input lists
entities_new, df_new1 = lemmatize(entities, df)
relationships_new, df_edge = lemmatize(relationships, df_new1)

# writing the edge list to a csv file. It wil be the base for RDF analysis
df.to_csv('rdf_list.csv', index=False, header=False)

edge_list_new = df_edge.values.tolist() # this is transforming the dataframe with the edges in a list of lists

# creating sentences from the edge list.
#   Each sentence has subject, predicates and object
sentences = []
for elem in edge_list_new:
    sentence = ' '.join(str(x) for x in elem)
    if sentence not in sentences:
        sentences.append(sentence)

# writing a csv file with the sentences
#print ('\n', sentences)
with open("knowledge_sentences.csv","w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(sentences)

# generating the graphs
graph_generation (edge_list_new)

print ('\n--- Graphs generated, RDF and Sentences files created - End of the process ---')

'''
The following is a sample of assistant and user prompts as strings
# prompts to create the relationships
text_user = """
A typical supply chain can be divided into two stages namely, production and distribution stages. 
In the production stage, components and semi-finished parts are produced in manufacturing centres. 
The components are then put together in an assembly plant. 
The distribution stage consists of central and regional distribution centres that transport products to end-consumers.[1]
At the end of the supply chain, materials and finished products only flow there because of the customer behaviour at the end of the chain;[7] 
academics Alan Harrison and Janet Godsell argue that "supply chain processes should be co-ordinated in order to focus on end customer buying behaviour", 
and look for "customer responsiveness" as an indicator confirming that materials are able to flow "through a sequence of supply chain processes in order to meet end customer buying behaviour".
"""
text_assistant = """
Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates. provide a json

If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.

Example:
prompt: Sun is source of solar energy. It is also source of Vitamin D.
updates:
[["Sun", "source of", "solar energy"],["Sun","source of", "Vitamin D"]]
"""
'''