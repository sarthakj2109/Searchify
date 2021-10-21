from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from pathlib import Path
import time
from haystack.document_store import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.pipeline import ExtractiveQAPipeline
  
class QnA:
  def __init__(self):
    self.document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    self.doc_dir="/content/drive/MyDrive/IR Project/PDF Files"
    self.dicts = convert_files_to_dicts(dir_path=self.doc_dir,split_paragraphs=True)
    self.document_store.write_documents(self.dicts)
    self.retriever = DensePassageRetriever(document_store=self.document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True)

    self.document_store.update_embeddings(self.retriever)
    self.reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    self.pipe = ExtractiveQAPipeline(self.reader, self.retriever)
  
  
  def predict(self,question):
    pdftext=[]
    offset=[]
    start_pred_time = time.time()
    print("In predict function")
    print(question)
    prediction = self.pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    end_pred_time = time.time()
    print_answers(prediction, details="minimal")
  # print(prediction)
    print(f'prediction time: {round(end_pred_time - start_pred_time, 2)}s')
    for i in range(5):
      offset.append((prediction['answers'][i].offsets_in_document[0].start,prediction['answers'][i].offsets_in_document[0].end))
      pdftext.append((self.document_store.get_document_by_id(prediction['answers'][i].document_id)).content)
    
    return prediction,pdftext,offset
