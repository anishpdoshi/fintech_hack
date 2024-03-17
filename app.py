import os, requests
from contextlib import ExitStack
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.api import partition_multiple_via_api

from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from query_llm import query_llm
import functools
import nltk
nltk.download('punkt')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO

import pandas as pd

TESTDATA = StringIO("""col1;col2;col3
    1;4.4;99
    2;4.5;200
    3;4.7;65
    4;3.2;140
    """)

df = pd.read_csv(TESTDATA, sep=";")

DATA_PATH = '/Users/apdoshi/mountain_view_pdfs/'
# DB_FAISS_PATH = 'vectorstore/db_faiss'

def get_pdf_paths():
    return [
        "/Users/apdoshi/mountain_view_pdfs/ex_pdf-part-2.pdf",
        # "/Users/apdoshi/mountain_view_pdfs/Fiscal Year 2016-17 Adopted Budget.pdf",
        # "/Users/apdoshi/mountain_view_pdfs/Fiscal Year 2017-18 Adopted Budget.pdf",
        # "/Users/apdoshi/mountain_view_pdfs/Fiscal Year 2019-20 Adopt.pdf",
        # "/Users/apdoshi/mountain_view_pdfs/Fiscal Year 2020-21 Adopt.pdf",
    ]

@functools.lru_cache()
def adapt_tables_to_common_schema(fname_to_tables):
    table_query = ''
    for fname, tables in fname_to_tables.items():
        print(type(tables))
        table_query += (fname.split('/')[-1] if '/' in fname else fname)
        table_query += '\n\n~~~~~~~~~~~~~~~\n\n'
        table_query += '\n'.join(table.text for table in tables)

    query = f"""
    Given the following table, give me a dataset summarizing the revenue, expenses, and net income
    broken down by granularity. For example, if the table is broken down by department, give me a dataset
    with the following columns: department, revenue, expenses, net income. Format the output in CSV format under the 
    header at the bottom of the prompt.
    
    {table_query[:2500]}

    Department/Bond Category;Revenue;Expenses;Net Income\n"""

    return query, query_llm(query, method="openai")

# Create vector database
@functools.lru_cache()
def query_pdfs_for_tables(pdf_paths):
    # loader = DirectoryLoader(DATA_PATH,
    #                          glob='*.pdf',
    #                          loader_cls=PyPDFLoader)
    res = {}
    for fname in pdf_paths:
        elements = partition_pdf(filename=fname,
                            infer_table_structure=True,
                            strategy='hi_res',
            )

        tables = [
            el for el in elements if el.category == "Table"
        ]

        res[fname] = tables
    return res

@functools.lru_cache()
def tables_as_data_frame(text):
    csv_text = text.split('\n\n')[:-1][0]
    TESTDATA = StringIO(f"""
        Department/Bond Category;Revenue;Expenses;Net Income
       {csv_text}
    """)

    df = pd.read_csv(TESTDATA, sep=";")
    return df

# Function to plot Revenue and Expenses per Department
def plot_revenue_expenses(df):
    # Plotting
    fig, ax = plt.subplots()
    # Width of a bar 
    width = 0.3

    # Set position of bar on X axis
    r1 = np.arange(len(df['Department']))
    r2 = [x + width for x in r1]

    # Make the plot
    ax.bar(r1, df['Revenue'], color='b', width=width, label='Revenue')
    ax.bar(r2, df['Expenses'], color='g', width=width, label='Expenses')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('Department', fontweight='bold')
    ax.set_xticks([r + width/2 for r in range(len(df['Department']))])
    ax.set_xticklabels(df['Department'])
    ax.set_ylabel('Amount ($)')
    ax.legend()
    ax.set_title('Revenue and Expenses per Department')

    return fig
        

if __name__ == "__main__":
    docs = query_pdfs_for_tables()
    query, res = adapt_tables_to_common_schema(docs)
    df = tables_as_data_frame(res)
    print(df)

    st.title('Financial Overview by Department')
    # Display DataFrame
    st.dataframe(df)

    # Generate and display plot
    fig = plot_revenue_expenses(df)
    st.pyplot(fig)