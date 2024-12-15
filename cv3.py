import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edges = pd.read_csv('KarateClub.csv',header=None,delimiter=';',names=['source','target'])

G = nx.from_pandas_edgelist(edges, 'source', 'target')

V_coefficients = pd.DataFrame(columns=['id', 'degree','closeness','coefficient'])

for node in G.nodes:

    clustering_coeff = nx.clustering(G, node)
    closeness = nx.closeness_centrality(G, u=node)

    V_coefficients.loc[len(V_coefficients)] = [node, 0, closeness, clustering_coeff]


#Coefficients
V_coefficients = V_coefficients.sort_values('id')

#Transitivity
transitivity = nx.transitivity(G)

degrees = pd.DataFrame(columns=['degree', 'coefficient','sum'])

for node in G.nodes:
    degree = G.degree[node]
    
    if degree not in degrees['degree'].values:
        
        coef = V_coefficients.loc[V_coefficients['id'] == node, 'coefficient'].values[0]
        V_coefficients.loc[V_coefficients['id'] == node, 'degree'] = degree

        degrees.loc[len(degrees)] = [degree,coef,1]
    else:
        coef = V_coefficients.loc[V_coefficients['id'] == node, 'coefficient'].values[0]
        degrees.loc[degrees['degree'] == degree, 'coefficient'] += coef
        degrees.loc[degrees['degree'] == degree, 'sum'] += 1

degrees = degrees.sort_values('degree')
V_coefficients.to_csv('result_cv3.csv',sep=';',index=False)

degrees['avg'] = degrees['coefficient'] / degrees['sum']

plt.figure(figsize=(8, 6))
plt.scatter(degrees['degree'], degrees['avg'], color='blue', marker='o')

plt.title('AVG/Degrees')
plt.xlabel('Degrees')
plt.ylabel('Average')

plt.grid()

plt.show()