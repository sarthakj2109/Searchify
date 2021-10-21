from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from pathlib import Path
import time
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipeline import ExtractiveQAPipeline
  
class QnA:
  def __init__(self):
    self.document_store=InMemoryDocumentStore()
    self.doc_dir="/content/drive/MyDrive/IR Project/PDF Files"
    self.dicts = convert_files_to_dicts(dir_path=self.doc_dir,split_paragraphs=True)
    self.document_store.write_documents(self.dicts)
    self.retriever = TfidfRetriever(document_store=self.document_store)
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
