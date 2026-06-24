#from langchain_huggingface import HuggingFaceEmbeddings
#from src.config import EMBEDDING_MODEL

#def get_embedding_model():
 #   return HuggingFaceEmbeddings(
  #      model_name=EMBEDDING_MODEL
   # )
from fastembed import TextEmbedding
from typing import List

class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(v) for v in vec] for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [float(v) for v in next(self.model.embed([text]))]


def get_embedding_model():
    return FastEmbedWrapper()
