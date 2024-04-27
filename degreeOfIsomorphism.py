import networkx as nx
SG = nx.read_gml("k_graph.gml")
CG = nx.read_gml("c_graph.gml")
SGNull = nx.empty_graph()
CGNull = nx.empty_graph()
print(nx.graph_edit_distance(SG, SGNull))
print(nx.graph_edit_distance(SG, CG))