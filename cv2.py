import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('KarateClub.csv',header=None,delimiter=';')

V = np.sort(pd.concat([df[0],df[1]]).unique())

distance_matrix = {}

#Create Adjency matrix/list
for vertex in V:
    distance_matrix[vertex] = {}    
    for u in V:
        distance_matrix[vertex].update({u:0})

#Change Adjency matrix to distance matrix(sets infinity, instied of 0)
for vertex in distance_matrix:
    for neighbour in distance_matrix[vertex]:
        if vertex != neighbour and len(df[((df[0] == vertex) & (df[1] == neighbour)) | ((df[0] == neighbour) & (df[1] == vertex))]):
            distance_matrix[vertex][neighbour] = 1
        elif vertex != neighbour and len(df[((df[0] == vertex) & (df[1] == neighbour)) | ((df[0] == neighbour) & (df[1] == vertex))]) == 0:
            distance_matrix[vertex][neighbour] = float('inf')

#Application of floyd algorithm
for k in distance_matrix:  
    for i in distance_matrix:  
        for j in distance_matrix:   
            if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

#Distance matrix with calculated shortest path between vertexes
distance_matrix_df = pd.DataFrame.from_dict(distance_matrix)

#Avarage distance for every vertex
distance_matrix_avg = distance_matrix_df.apply(lambda row: row[row != 0].mean(),axis=1)
vertexes_edges_avg = pd.DataFrame(list(distance_matrix_avg.items()), columns=['Vertex', 'avg'])

#Avarage distance
avg_distance = vertexes_edges_avg['avg'].mean()

#Graph diameter
graph_diameter = max(distance_matrix_df.max(axis=1))

#Clossness
distance_matrix_rows_sum = distance_matrix_df.sum(axis = 1)
closeness = pd.DataFrame(list(distance_matrix_rows_sum.items()), columns=['Vertex', 'sum'])

print(closeness)
closeness['closeness'] = len(V) / closeness['sum']