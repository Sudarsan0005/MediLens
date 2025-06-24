from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel
from src.rag_workflow.custom_exception.custom_exception import ChunkException

class DataExtractor(BaseModel):
    def __init__(self,output_dir:str="documents"):
        self.output_dir=output_dir
        self.tables=[]
        self.text=[]
        self.images=[]
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

    def extract_table(self,chunks):
        try:
            for chunk in chunks:
                if "Table" in str(type(chunk)):
                    self.tables.append(chunk)
        except:
            raise ChunkException("Error while extracting tables from chunks")


