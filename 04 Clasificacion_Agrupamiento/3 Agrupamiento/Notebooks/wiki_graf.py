######################################### 
# Funciones auxiliares para el 
# procesamiento de texto de Wikipedia
#
# autor: Jorge Hermosillo
# Fecha: 22-sep-2019
# curso: Escuela de Ciencia de Datos 2019
##########################################
import sys
import string
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#####################################
#  graficación de palabras por doc  #
#Generamos los vectores que vamos a # 
# usar en los gráficos de barras:   #
# * x: contiene la enumeración      #
#      de los documentos            #
# * y: contiene sus respectivos     #
#       totales de palabras         #
#####################################
def grafica_palabras_porDoc(datos,nombre='barras',ancho=0.8):
    #ordena los datos en orden descendente y saca el promedio.
    promedio = datos['Total'].mean()
    print('Promedio de palabras por documento en el corpus: {}'.format(promedio))
    
    #obtiene los valores de x y y
    x=np.arange(len(datos.index.values))
    etiquetas=[]
    
    for e in datos.index.values:
        etiquetas.append(str(e))
    
    y=datos['Total'].values
    print(y[:10])
    
    #define el área de dibujo
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_facecolor('white')
    plt.grid(False)
    ax.tick_params(axis='x', labelsize=6)

    #graficación
    #ancho = 0.8 #ancho de las barras
    ax.bar(x - ancho/2, y, ancho, label='Totales')
    ax.axhline(y=promedio, \
               color='r', \
               linestyle='--', \
               label='Promedio')

    # Etiquetas, títulos, etc.
    ax.set_ylabel('Palabras')
    ax.set_title('Número de palabras por documento')
    ax.set_xticks(x)
    plt.xticks(rotation=90)
    ax.set_xticklabels(etiquetas)
    ax.legend()
    #plt.savefig('img/'+nombre+'.pdf')
    plt.show()
    return

def grafica_docs(df,titulo='Documentos'):
    #""" Obtención de valores"""
    docs_0=df[df.clase==0]
    docs_1=df[df.clase==1]

    #"""Areas de Graficacion y visualizacion de los datos"""
    fig,ax = plt.subplots(figsize=(5,5))

    #"""Documentos en clase 0"""
    ax.scatter(docs_0.c0, docs_0.c1,
               facecolor='royalblue', 
               marker='o', 
               edgecolor='blue',
               s=20,
               alpha=0.5,
               label='Docs_0')

    #"""Documentos en clase 1"""
    ax.scatter(docs_1.c0, docs_1.c1,
               facecolor='orangered', 
               marker='o', 
               edgecolor='red',
               s=20,
               alpha=0.5,
               label='Docs_1')
    plt.title(titulo)
    plt.xlabel('c0')
    plt.ylabel('c1')
    ax.legend()
    return ax


def palabras_comunes(df):
    Palabras=df['Palabras'].values.tolist()
    docs = df.doc_id.values
    lista=[]
    for i in range(len(Palabras)):
        lista.append((docs[i],Palabras[i]))
    palco=[]
    nopalco=[]
    for i in range(len(lista)):
        for j in range(i+1,len(lista)):
            palco.append(((lista[i][0],lista[j][0]),\
                          lista[i][1] & lista[j][1]))
            nopalco.append(((lista[i][0],lista[j][0]),\
                            lista[i][1] | lista[j][1] - \
                            lista[i][1] & lista[j][1]))

    palco=sorted(palco,key=lambda x: len(x[1]),reverse=True)
    nopalco=sorted(nopalco,key=lambda x: len(x[1]),reverse=True)
    npd = pd.DataFrame(nopalco).drop(columns=[0])
    paldoc=pd.DataFrame(palco)
    paldoc=pd.concat([paldoc,npd],ignore_index=True, sort=False,axis=1)
    paldoc.columns=['_ids','PalCom','PalNoCom']
    return paldoc
