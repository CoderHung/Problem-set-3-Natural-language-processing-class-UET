import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import words
import re

#nltk.download('words')
english_words = set(words.words())

def read_text_files(directory):
    documents = []
    filenames = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                words = [word for word in re.findall(r'\b\w+\b', text.lower()) if word in english_words]
                words = ' '.join(words)
                documents.append(words)
                filenames.append(filename)
    
    filenames = sorted(filenames, key=lambda x: int(x.split(' - ')[0]))
    filenames = [entry.split(' - ')[1] for entry in filenames]
    return documents, filenames


def create_term_document_matrix(documents):
    vectorizer = CountVectorizer(strip_accents='unicode',min_df = 0.03,stop_words='english')  
    tdm = vectorizer.fit_transform(documents).toarray()
    tdm = np.transpose(tdm)
    return tdm, vectorizer.get_feature_names_out()

directory = 'Data'  
documents, filenames = read_text_files(directory)
tdm, terms = create_term_document_matrix(documents)
tdm_df = pd.DataFrame(tdm ,columns=filenames, index=terms)
print(tdm_df.info)
tdm_df.to_csv('TD.csv')
