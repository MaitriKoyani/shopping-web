from flask import Flask,render_template,request
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

app = Flask(__name__)

model=SentenceTransformer('all-mpnet-base-v2')
# model=SentenceTransformer('all-MiniLM-L6-v2')
sentences=[]
maindata=[]
with open('data.json', 'r') as f:
    data=f.read()
    data=json.loads(data)
    for key in data:
        for ke in data[key]:
            des={ke:data[key][ke]}
            oc=''
            d=''
            try:
                if data[key][ke]['Occasion']:
                    oc=data[key][ke]['Occasion']
            except Exception as e:
                oc=''
            if data[key][ke]['description']:
                d=data[key][ke]['description']
            
            mai=oc+" and "+d
            sentences.append(mai)
            maindata.append(des)
            
text_embeddings = model.encode(sentences, convert_to_tensor=True)
text_embeddings=np.asarray(text_embeddings)
faiss.normalize_L2(text_embeddings)
d=text_embeddings.shape[1]
print(text_embeddings.shape)
index=faiss.IndexFlatL2(d)
index.add(text_embeddings)
contexts = {
    "formal": ["formal","interview", "professional", "meeting", "office", "corporate","bussiness","award"],
    "party": ["party", "celebration","festival","date"],
    "wedding-traditional-indian-saree": ["wedding", "festival","family dinner","marriage"],
    "casual-cords-top": ["casual", "daily", "relaxed", "weekend","family","walking","summer"],
    "sports": ["gym", "sports", "workout", "athletic", "running"],
    "travel-jumpsuit": ["travel", "trip", "vacation", "outdoor"],
    "nightsuit": ["nightsuit","sleep", "home", "indoor"],
    "sweatshirt": ["winter","sweatshirt"]
}
def infer_context(query, contexts):
    query_lower = query.lower()
    for context, keywords in contexts.items():
        print(keywords,query_lower)
        if any(keyword in query_lower for keyword in keywords):
            print('enter')
            print(context)
            return context
    context='Normal' 
    return context
def find(query):
    context = infer_context(query, contexts)
    print(context)
    query_embedding = model.encode(query, convert_to_tensor=True)
    context_embedding = model.encode(context, convert_to_tensor=True)
    combined_embedding = 0.5 * query_embedding + 0.5 * context_embedding
    query_embedding = np.asarray(combined_embedding).reshape(1, -1)
    # query_embedding = np.asarray(query_embedding).reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    k=12
    distances, indices = index.search(query_embedding, k)
    
    products=[]
    for i, idx in enumerate(indices[0]):
        new={}
        for m in maindata[idx].values():
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