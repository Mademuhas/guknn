import os

import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import logging

from sklearn.metrics import (f1_score,
                            accuracy_score,
                            classification_report,
                            recall_score,
                            precision_score,
                            confusion_matrix,
                            roc_auc_score)

from imblearn.under_sampling import RandomUnderSampler

import faiss
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

@app.route('/comply', methods=['POST'])
def comply():
    data = request.get_json(force=True)
    documentos = data['documentos']
    bases = data['bases']

    textos = [documento['text'] for documento in documentos]
    titulos = [documento['subtitle'] for documento in documentos]
    base = [base['text'] for base in bases]
    output_base = [base['comply'] for base in bases]
    output_amostra = []

    concat_amostras = [f"{texto}:{titulo}" for texto, titulo in zip(textos, titulos)]

    model = SentenceTransformer('intfloat/multilingual-e5-large')
    lista_knn = []
    lista_knn_total = []
    lista_embedding = []
    for i in range (len(concat_amostras)):
        embedding_pm = model.encode(f'{concat_amostras[i]}')
        lista_embedding.append(embedding_pm)
        
        
    index = faiss.IndexFlatL2(len(lista_embedding[0]))
    index.is_trained
    matrix = np.zeros ( (len(lista_embedding), len(lista_embedding[0]) ) )
    for i in range (matrix.shape[0]):
        matrix[i][:] = lista_embedding[i]
    index.add(matrix)
    
    k = 5
 
    for i in range (len(base)):
        texto = f'{base[i]}'
        xq = model.encode([texto])
        D, I = index.search(xq, k)
        lista_knn = I[0]
        lista_knn_total.append(lista_knn)
        lista_knn = []

    for j, knn in enumerate(lista_knn_total):
        count_ok = 0
        count_c = 0
        count_d = 0
        for i in range(5):
            indice = lista_knn_total[i][j]
            if output_base[indice] == 'OK':
                count_ok +=1
            elif output_base[indice] == 'C':
                count_c +=1
            elif output_base[indice] == 'D':
                count_d +=1
    if (count_ok > count_c) and (count_ok >= count_d):
        output_amostra.append('OK')
    elif (count_c >= count_ok) and (count_c > count_d):
        output_amostra.append('C')
    elif (count_c > count_ok) and (count_c >= count_d):
        output_amostra.append('C')
    elif (count_d > count_ok) and (count_d > count_c):
        output_amostra.append('D')
    print(output_amostra)

if __name__ == '__main__':
    app.run(debug=True)







