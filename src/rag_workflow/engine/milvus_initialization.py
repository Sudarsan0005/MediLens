import logging
import os
from embeddings_generator import hf_embd
from langchain_milvus import Milvus
from pymilvus import MilvusClient, DataType,FunctionType,Function,AnnSearchRequest
from dotenv import load_dotenv
from pymilvus import RRFRanker

load_dotenv()

class MilvusDB:
    def __init__(self):
        self.SERVER_ADDR = "http://localhost:19530"
        self.client = MilvusClient(uri=self.SERVER_ADDR)
        self.schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
        )
        self.index_params = self.client.prepare_index_params()
        self.bm25_function = Function(
                    name="text_bm25_emb",
                    input_field_names=["Summary"],
                    output_field_names=["sparse_vector"],
                    function_type=FunctionType.BM25,
                )
        self.ranker = RRFRanker(100)
    async def createtext_schema(self,collection_name):
        try:
            if not self.client.has_collection(collection_name):
                self.schema.add_field(field_name="id",datatype=DataType.VARCHAR,max_length=100,is_primary=True)
                self.schema.add_field(field_name="Text", datatype=DataType.VARCHAR, max_length=15000)
                self.schema.add_field(field_name="Summary", datatype=DataType.VARCHAR, max_length=10000)
                self.schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=384)
                self.schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
                self.schema.add_field(field_name="Document_name", datatype=DataType.VARCHAR, max_length=100)
                self.schema.add_field(field_name="Page_no", datatype=DataType.VARCHAR, max_length=50)

                self.schema.add_function(self.bm25_function)
                self.index_params.add_index(
                    field_name="dense_vector",
                    index_name="dense_vector_index",
                    index_type="AUTOINDEX",
                    metric_type="COSINE"
                )
                self.index_params.add_index(
                    field_name="sparse_vector",
                    index_name="sparse_inverted_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                    params={"inverted_index_algo": "DAAT_MAXSCORE"},  # or "DAAT_WAND" or "TAAT_NAIVE"
                )
                self.client.create_collection(
                    collection_name=collection_name,
                    schema=self.schema,
                    index_params=self.index_params
                )
                logging.info(f"Collection: {collection_name} created")
            else:
                logging.info(f"Collection: {collection_name} already exists.")
        except Exception as e:
            raise Exception(f"Error while creating vector schema{e}")

    async def createtable_schema(self,collection_name):
        try:
            if not self.client.has_collection(collection_name):
                self.schema.add_field(field_name="id",datatype=DataType.VARCHAR,max_length=100,is_primary=True)
                self.schema.add_field(field_name="Table_raw", datatype=DataType.VARCHAR, max_length=9000)
                self.schema.add_field(field_name="Summary", datatype=DataType.VARCHAR, max_length=8000)
                self.schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=384)
                self.schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
                self.schema.add_field(field_name="Document_name", datatype=DataType.VARCHAR, max_length=100)
                self.schema.add_field(field_name="Page_no", datatype=DataType.VARCHAR, max_length=50)
                self.schema.add_function(self.bm25_function)
                self.index_params.add_index(
                    field_name="dense_vector",
                    index_name="dense_vector_index",
                    index_type="AUTOINDEX",
                    metric_type="COSINE"
                )
                self.index_params.add_index(
                    field_name="sparse_vector",
                    index_name="sparse_inverted_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                    params={"inverted_index_algo": "DAAT_MAXSCORE"},  # or "DAAT_WAND" or "TAAT_NAIVE"
                )
                self.client.create_collection(
                    collection_name=collection_name,
                    schema=self.schema,
                    index_params=self.index_params
                )
                logging.info(f"Collection: {collection_name} created")
            else:
                logging.info(f"Collection: {collection_name} already exists.")
        except Exception as e:
            raise Exception(f"Error while creating vector schema{e}")

    async def createimage_schema(self,collection_name):
        try:
            if not self.client.has_collection(collection_name):
                self.schema.add_field(field_name="id",datatype=DataType.VARCHAR,max_length=100,is_primary=True)
                self.schema.add_field(field_name="image_base64", datatype=DataType.VARCHAR, max_length=65000)
                self.schema.add_field(field_name="Summary", datatype=DataType.VARCHAR, max_length=8000)
                self.schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=384)
                self.schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
                self.schema.add_field(field_name="Document_name", datatype=DataType.VARCHAR, max_length=100)
                self.schema.add_field(field_name="Page_no", datatype=DataType.VARCHAR, max_length=50)
                self.schema.add_function(self.bm25_function)
                self.index_params.add_index(
                    field_name="dense_vector",
                    index_name="dense_vector_index",
                    index_type="AUTOINDEX",
                    metric_type="COSINE"
                )
                self.index_params.add_index(
                    field_name="sparse_vector",
                    index_name="sparse_inverted_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                    params={"inverted_index_algo": "DAAT_MAXSCORE"},  # or "DAAT_WAND" or "TAAT_NAIVE"
                )
                self.client.create_collection(
                    collection_name=collection_name,
                    schema=self.schema,
                    index_params=self.index_params
                )
                logging.info(f"Collection: {collection_name} created")
            else:
                logging.info(f"Collection: {collection_name} already exists.")
        except Exception as e:
            raise Exception(f"Error while creating vector schema{e}")

    async def insert_collection(self,data,collection_name):
        try:
            if self.client.has_collection(collection_name):
                res = self.client.insert(
                    collection_name=collection_name,
                    data=data
                )
                return res
            else:
                raise Exception(f"vector collection - {collection_name} doesn't exist")
        except Exception as e:
            raise Exception(f"Error while inserting vector collection - {collection_name}, Error- {e}")

    async def retrive_data(self,query_text,query_dense_vector,collection_name):
        try:
            search_param_1 = {
                "data": [query_dense_vector],
                "anns_field": "text_dense",
                "param": {"nprobe": 10},
                "limit": 2
            }
            request_1 = AnnSearchRequest(**search_param_1)

            # full-text search (sparse)
            search_param_2 = {
                "data": [query_text],
                "anns_field": "text_sparse",
                "param": {"drop_ratio_search": 0.2},
                "limit": 2
            }
            request_2 = AnnSearchRequest(**search_param_2)
            reqs = [request_1,request_2]
            res = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=self.ranker,
                limit=2
            )
            return res
        except Exception as e:
            logging.critical(f"Error while retriving data from vector database {e}")

