from flask import Flask,render_template,request
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

app = Flask(__name__)

model=SentenceTransformer('all-mpnet-base-v2')
sentences=[]
with open('data.json', 'r') as f:
    data=f.read()
    data=json.loads(data)
    for key in data:
        for ke in data[key]:
            des={ke:{}}
            for k in data[key][ke]:
                des[ke].update({k : data[key][ke][k]})
                
            sentences.append(des)
text_embeddings = model.encode(sentences, convert_to_tensor=True)
text_embeddings=np.array(text_embeddings)
faiss.normalize_L2(text_embeddings)
d=text_embeddings.shape[1]
print(text_embeddings.shape)
index=faiss.IndexFlatL2(d)
index.add(text_embeddings)
def find(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
                          
    query_embedding = np.array(query_embedding).reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    k=50
    distances, indices = index.search(query_embedding, k)
    
    products=[]
    for i, idx in enumerate(indices[0]):
        new={}
        for m in sentences[idx].values():
            for k,j in m.items():
                new.update({k:j})
                
        products.append({idx:new})
    return products

@app.route("/")
def indexo():
    return render_template('index.html')

@app.route("/search",methods=['GET','POST'])
def search():
    if request.method == 'POST':
        if request.form['search']:
            query = request.form['search']
            products=find(query)
            
            return render_template('index.html',products=products,query=query)
if __name__ == "__main__":
    app.run(debug=True)