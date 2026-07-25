from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

def document_loader(chunk_methods):

  documents = [
    r"..\document\SK_REKTOR_70_SK_Rektor_Tentang_Bentuk_Tugas.pdf",
    r"..\document\SK_REKTOR_1221_Pedoman_pengenalan_budaya_akad.pdf",
    r"..\document\SK_REKTOR_1280_Pedoman_Akademik_PROGRAM_SARJANA.pdf"
  ]

  raw_doc = []
  for doc in documents:
    loader = PyPDFLoader(doc)
    raw_doc.extend(loader.load())

  match chunk_methods:
    case "fixed-chunk":
      splitter = CharacterTextSplitter(
        separator = " ",
        chunk_size = 256,
        chunk_overlap = 20
      )
    
    case "recursive-chunk":
      splitter = RecursiveCharacterTextSplitter(
        chunk_size = 256,
        chunk_overlap = 20,
        separators = ["\n\n", "\n", " ", ""]
      )

    case "semantic-chunk":
      embeddings = OllamaEmbeddings(model = "bge-m3")
      splitter = SemanticChunker(embeddings)
    
  chunked_documents = splitter.split_documents(raw_doc)

  return chunked_documents