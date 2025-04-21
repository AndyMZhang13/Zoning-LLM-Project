HOW TO RUN THE CODE:

SETUP DEPENDANCIES
# pip install pymupdf faiss-cpu sentence-transformers transformers torch tqdm protobuf==3.20.3 keras==2.11.0 tensorflow==2.11.0

Replace pdf_path with path to pdf (right click copy path, then fix formatting)
# EXAMPLE: pdf_path = "C:/Users/EXAMPLEUSER/.vscode/SOMEFOLDER/test.pdf" 


Run zoning_code.py to generate embeddings and text chunks
Then, run query_faiss_local.py to query the locally generated files

KNOWN ISSUES AND FUTURE IMPROVEMENTS:
- You must manually replace in the code the pdf path
- You must clean artifacts of old pdfs if you want to use it on a new pdf file

  TODOS:
  
