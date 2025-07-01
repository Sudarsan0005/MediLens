from data_extractor import DataExtractor
from milvus_initialization import MilvusDB
from embeddings_generator import hf_embd

class Data_manager:
    def __init__(self):
        self.data_extractor=DataExtractor()
        self.vector_db_manager = MilvusDB()
    async def _insert_to_DB(self,file_path:str=None):
        if file_path.split(".")[-1]=="pdf":
            data_chunks = await self.data_extractor.make_chunks(file_path=file_path)
            for data in data_chunks:
                data = data.to_dict()
                data_type = data.get("type","")
                text = data.get("text","")
                page_no = data.get("page_number","")

