from langchain.retrievers.multi_vector import MultiVectorRetriever

class ChunkException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"