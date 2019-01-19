import os
import json

class WikiDataLoader:
    def __init__(self, path, jsonf=False):
        self.jsonf = jsonf

        # Path to extracted wiki dump
        self.path = path
        # current file object in extracted wiki dump
        self.cFileObj = None

        self.listSubfolders()
        # List of documents from wiki dump file
        self.docs = None

        self.cSub = self.subfolders.pop(0)
        self.cFile = None

        self.getFilesFromCurrentSubfolder()

    def listSubfolders(self):
        self.subfolders = list(filter(lambda x: os.path.isdir(os.path.join(self.path,x)), os.listdir(self.path)))
        self.subfolders.sort() # Ensure alphabetical order

    def getFilesFromCurrentSubfolder(self):
        sub_path = os.path.join(self.path,self.cSub)
        self.files = list(filter(lambda x: os.path.isfile(os.path.join(sub_path,x)), os.listdir(sub_path)))
        self.files.sort()

    def next_file(self):

        if not self.files:

            if not self.subfolders:
                self.cFileObj = None
                return
                
            self.cSub = self.subfolders.pop(0)
            self.getFilesFromCurrentSubfolder
            # sub_path = os.path.join(self.path, self.subfolders[self.subfolder_pos])
            # self.files = list(filter(lambda x: os.path.isfile(os.path.join(sub_path,x)), os.listdir(sub_path)))
            # self.files.sort()

        self.cFile = self.files.pop(0)

        # path to the current file
        file_path = os.path.join(os.path.join(self.path, self.cSub),self.cFile)
        
        # If a file was opened, close it
        if self.cFileObj is not None:
            self.cFileObj.close()
        
        self.cFileObj = open(file_path, "r")


    def load_new_docs(self):
        self.next_file()

        # When no more files available, return None
        if self.cFileObj is None:
            self.docs = None
            return

        if self.jsonf:
            docs = self.cFileObj.read().strip().split("\n")
            docs_list = [json.loads(doc)['text'] for doc in docs]
        else:
            docs = self.cFileObj.read().split("</doc>")
            docs_list = []

            for doc in docs:
                if doc.strip():
                    # filter the first line that contains <doc> tag
                    docs_list.append("\n".join(doc.split("\n")[1:]))

        self.docs = docs_list

    def next_doc(self):
        '''
        Return next available document
        '''

        if not self.docs:
            self.load_new_docs()

        if self.docs:
            return self.docs.pop(0)
        else:
            return None
