# Searchify
Searchify is an easy to use robust search tool that allows user to query in Natural Language. This project is based on [haystack](https://github.com/deepset-ai/haystack) (2.6K stars on GitHub), an open source end-to-end python framework that uses state of the art NLP models to develop Question-Answering systems.  

### Steps to run the tool:
1) Clone this repository and download the Searchify Folder. Upload this folder on your personal Google Drive.
2) You can specify a corpus of your choice, simply replace the files in Corpus folder with your desired files. 
The corpus supports the following file types : Txt, PDF (incl. OCR), Docx, Apache Tika (Supports > 340 file formats), Markdown and Images
3) In the searchify.ipynb file, change the path according to the location on your drive
4) Run all the cells of the .ipynb (Preferably use Google Colab software as you'll require GPU support for faster execution)
5) Click on the public url generated as part of the output. (For instance:  ngrok tunnel "http://9cb2-34-69-188-223.ngrok.io")
6) And voila! you can now use the Searchify tool. Simply enter what you wish to search for and click on the 'Search' button.

### Technologies and Algorithms used:

-> The server.py uses InMemory Document Store and TF-IDF Retriever to retrieve the most probable answers to the given query

-> The server2.py uses FAISS (Facebook Similarity Search) Document Store and Dense Passage Retriever (DPR) 

-> The ROBERTA model developed by Fairseq is used as the reader

-> The tool is deployed using Flask framework in Python

-> The web app is designed using HTML, CSS and JS

NOTE: You can refer to the Presentation.pptx for overview of the project and screenshots of the tool in action!
