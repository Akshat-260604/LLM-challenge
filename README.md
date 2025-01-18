  LAST UPDATED - 18TH JAN.
  
Will keep it very short. I have tried to make a chatbot that takes in query from a book named - Farmerbook. This book consists of topics related to Agriculture. 
For the very first time, I used the pinecone vector database. The alteranative to this is FAISS...that I used in one of my other projects. 
The workflow of the model in short is as follows:- 

- Initialization of HF model
- Query
- Query Embedding
- Pinecone initalization
- Extract texts from PDF
- preprocess and clean text
- query match and track page reference
- process query
- context combination
- get answer using qa model
- display answer

MOST OF THE CODE IS CRAP.
