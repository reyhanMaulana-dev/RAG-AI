from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class ChromaDatabase:
  
  def __init__(self, persist_directory, embedding_model):
    self.persist_directory = persist_directory
    self.embeddings = OllamaEmbeddings(model = embedding_model)

  def get_vector_db(self, collection_name):
    return Chroma (
      collection_name = collection_name,
      persist_directory = self.persist_directory,
      embedding_function = self.embeddings
    )
  
  def get_retriever(self, collection_name):
    vector_db = self.get_vector_db(collection_name)
    retriever = vector_db.as_retriever(search_kwargs = {"k" : 3})

    return retriever
  
  def add_document(self, collection_name, documents):
    vector_db = self.get_vector_db(collection_name)
    vector_db.add_documents(documents = documents)
    print(f"Berhasil menambahkan {len(documents)} dokumen ke koleksi '{collection_name}'")

  def delete_collection(self, collection_name):
    vector_db = self.get_vector_db(collection_name)

    try:
      vector_db._client.delete_collection(name = collection_name)
      print(f"Koleksi {collection_name} telah dihapus secara permanen")
    except Exception as e:
      print(f"Gagal menghapus koleksi {collection_name} : {e}")

vector_db = ChromaDatabase(
  persist_directory = "../chroma_db",
  embedding_model = "bge-m3"
)