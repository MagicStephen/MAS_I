import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn 

df = pd.read_csv('KarateClub.csv',header=None,delimiter=';')

vertexes = np.sort(pd.concat([df[0],df[1]]).unique())
edges = {}

adjency_matrix = {}
adjency_list = {}

#Create Adjency matrix/list
for v in vertexes:
    adjency_matrix[v] = {}
    adjency_list[v] = []
    for u in vertexes:
        adjency_matrix[v].update({u:0})

#Searching throught all vertexes
for v in adjency_matrix:
    for neighbor in adjency_matrix[v]:
        if v != neighbor and len(df[((df[0] == v) & (df[1] == neighbor)) | ((df[0] == neighbor) & (df[1] == v))]):
            adjency_matrix[v][neighbor] = 1
            adjency_list[v].append(neighbor)

#Create Dataframe of matriy
adjency_matrix_df = pd.DataFrame.from_dict(adjency_matrix, orient='index')
adjency_matrix_rows_sum = adjency_matrix_df.sum(axis = 1)

#Get from degrees(max,min,avg)
adjency_matrix_max = adjency_matrix_rows_sum.max()
adjency_matrix_min = adjency_matrix_rows_sum.min()
adjency_matrix_avg = adjency_matrix_rows_sum.mean()

#Get frequencies(relative,normal) from matrix
frequency_of_matrix = adjency_matrix_rows_sum.value_counts()
relative_frequency_of_matrix = adjency_matrix_rows_sum.value_counts(normalize=True)

#Create Dataframe of list
adjency_list_df = pd.DataFrame(list(adjency_list.items()), columns=['Vertex', 'Neighbors'])
adjency_list_df['degree'] = adjency_list_df['Neighbors'].apply(len)

#Get from degrees(max,min,avg)
adjency_list_max = adjency_list_df['degree'].max()
adjency_list_min = adjency_list_df['degree'].min()
adjency_list_avg = adjency_list_df['degree'].mean()

#Get frequencies(relative,normal) from list
frequency_of_list = adjency_list_df['degree'].value_counts()
relative_frequency_of_list = adjency_list_df['degree'].value_counts(normalize=True)

#Change series to dataframe
degree_df = frequency_of_list.reset_index()
degree_df.columns = ['Degree', 'Count']

#Create chart with Degree(x) and Nodes(y)
bins_x = range(1,degree_df['Degree'].max()+1)
bins_y = range(1,12)

#Create histogram
sn.histplot(data=degree_df, x='Degree', weights='Count', bins=bins_x, discrete=True)
plt.xticks(bins_x)
plt.yticks(bins_y)
plt.axvline(x=adjency_list_avg, color='red', linestyle='--',label="Average "+str(adjency_list_avg.round(2))+' Degrees')
plt.xlabel('Degrees(# of connections)')
plt.ylabel('Nodes(# count)')
plt.title('Karate Club Degree Distribution')
plt.legend()
plt.show()





