import os, requests
from contextlib import ExitStack
import streamlit as st
import pandas as pd
import numpy as np

from fpdf import FPDF
from io import StringIO
import io
import matplotlib.pyplot as plt
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.api import partition_multiple_via_api

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from query_llm import query_llm

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import functools
import nltk
nltk.download('punkt')

def get_pdf_paths():
    return tuple([
        "/Users/apdoshi/mountain_view_pdfs/extracted_pdfs/Fiscal Year 2016-17 Adopted Budget Extracted.pdf",
        "/Users/apdoshi/mountain_view_pdfs/extracted_pdfs/Fiscal Year 2017-18 Adopted Budget Extracted.pdf",
        "/Users/apdoshi/mountain_view_pdfs/extracted_pdfs/Fiscal Year 2019-20 Adopt Extracted.pdf",
        "/Users/apdoshi/mountain_view_pdfs/extracted_pdfs/Fiscal Year 2020-21 Adopt Extracted.pdf",
        "/Users/apdoshi/mountain_view_pdfs/extracted_pdfs/Fiscal Year 2022-2023 Budget Extracted.pdf"
    ])

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

def get_insights(fname_to_tables):
    table_query = ''
    for fname, tables in fname_to_tables.items():
        table_query += (fname.split('/')[-1] if '/' in fname else fname)
        table_query += '\n\n~~~~~~~~~~~~~~~\n\n'
        table_query += '\n'.join(table.text for table in tables)


    return query, query_llm(
        """
        Please give me insights in markdown format, including respective financial data that will be relevant for a bond trader who is looking to price and purchase bonds from this municipality using the below text.

        Important: Use the right scale for the numbers as highlighted in the text and positive or negative values accordingly.

        Here is the tables per year:
        {table_query[:2500]}
        """,
        method='openai'
    )


# Function to plot Revenue and Expenses per Department
def plot_revenue_expenses(df):
    # Plotting
    fig, ax = plt.subplots()
    # Width of a bar 
    width = 0.3

    # Set position of bar on X axis
    r1 = np.arange(len(df))
    r2 = [x + width for x in r1]

    # Make the plot
    ax.bar(r1, df['Revenue'], color='b', width=width, label='Revenue')
    ax.bar(r2, df['Expenses'], color='g', width=width, label='Expenses')

    # Add xticks on the middle of the group bars
    ax.set_xlabel(df.columns[0], fontweight='bold')
    ax.set_xticks([r + width/2 for r in range(len(df[df.columns[0]]))])
    ax.set_xticklabels(df[df.columns[0]])
    ax.set_ylabel('Amount ($)')
    ax.legend()
    ax.set_title('Revenue and Expenses per Department')

    return fig
        
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


if __name__ == "__main__":
    docs = query_pdfs_for_tables(get_pdf_paths())
    query, res = adapt_tables_to_common_schema(docs)
    insight_query, insights_text = get_insights(docs)

    df = tables_as_data_frame(res)

    # df.style.highlight_max(axis=0)
    
    # Sort and select the top 10 categories based on user's choice
    sort_option = st.radio("Sort by:", ('Revenue', 'Expenses'))
    df_sorted = df.sort_values(by=sort_option, ascending=False).head(10)


    st.title('Financial Overview by Department')
    # Display DataFrame
    st.dataframe(df_sorted)

    st.markdown(insights_text)

    st.text_input("Enter your query here (e.g. 'What is information I need to price bonds accurately?)'")

    #  Generate and display plot
    fig = plot_revenue_expenses(df_sorted)
    st.pyplot(fig)

    export_as_pdf = st.button("Export Report")

    if export_as_pdf:
        # Export functionality as before
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_image = Image(buf, width=6*inch, height=4*inch)  # Adjust size as needed

        pdf_filename = 'report.pdf'
        doc = SimpleDocTemplate(pdf_filename)
        styles = getSampleStyleSheet()

        csv_style = ParagraphStyle(name='CSVStyle', parent=styles['Code'], fontSize=8, leading=8.5)
        story = [
            Paragraph("Financial Overview by Department (Top 10)", styles["h1"]),
            Spacer(1, 0.2 * inch),
            Preformatted(df_sorted.to_string(index=False), csv_style),
            Spacer(1, 0.2 * inch),
            Paragraph(insights_text, styles["Normal"]),
            Spacer(1, 0.2 * inch),
            plot_image
        ]

        doc.build(story)

        with open(pdf_filename, "rb") as file:
            st.download_button(
                label="Download PDF Report",
                data=file,
                file_name=pdf_filename,
                mime="application/pdf"
            )