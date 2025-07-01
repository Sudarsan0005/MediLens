from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel
from src.rag_workflow.custom_exception.custom_exception import ChunkException
from typing import List

class DataExtractor(BaseModel):
    def __init__(self,output_dir:str="documents"):
        self.output_dir=output_dir
        self.tables=[]
        self.text=[]
        self.images_b64=[]
    def make_chunks(self,file_path:str=None):
        try:
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,  # extract tables
                strategy="hi_res",  # mandatory to infer tables

                extract_image_block_types=["Image"],
                # image_output_dir_path=output_path,
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
                # extract_images_in_pdf=True,          # deprecated
            )
            return chunks
        except:
            raise ChunkException("Error while making chunks with unstructured")

    def extract_table(self,chunks:List=None):
        try:
            for chunk in chunks:
                if "Table" in str(type(chunk)):
                    self.tables.append(chunk)
            return self.tables
        except:
            raise ChunkException("Error while extracting tables from chunks")

    def extract_text(self,chunks:List=None):
        try:
            for chunk in chunks:
                if "CompositeElement" in str(type((chunk))):
                    self.text.append(chunk)
            return self.text
        except:
            raise ChunkException("Error while extracting text from chunks")
    def extract_image(self,chunks:List=None):
        try:
            for chunk in chunks:
                if "CompositeElement" in str(type(chunk)):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            self.images_b64.append(el.metadata.image_base64)
            return self.images_b64
        except:
            raise ChunkException("Error while extracting images from chunks")



