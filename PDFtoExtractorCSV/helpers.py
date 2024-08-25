from langchain_openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import openai
import re
import os
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
import pandas as pd
from pypdf import PdfReader
import ast
from openai import OpenAI

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(temperature=0.7)

client = OpenAI()

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extracted_data(pages_data):
    template = """Extract the following values from what will be given:
    Date (DO NOT INCLUDE STARTING 0s to the data, so 01/05/2024 should be 1/05/2024),
    Details (ONLY the EVR tag e.g EVR292544),
    Parcels,
    Net Amount,
    Vat Amount. Extract these values from the following text:

    {pages}
    
    Expected Output: remove any symbols and dots, do not miss a page, we should take in all values, please do not truncate the output and continue for as long as you can e.g Date: 1/05/2024, Details: EVR292544, Parcels: 1, Net Amount: £3, Vat Amount: £0.20
    """
    
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    formatted_prompt = prompt_template.format(pages=pages_data)

    stream = client.chat.completions.create(
        model="chatgpt-4o-latest", 
        messages=[{"role": "user", "content": formatted_prompt}],
        stream=True,
        max_tokens=16384  # Adjust as needed
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content

    return full_response

def parse_extracted_text(extracted_text):
    entries = extracted_text.split('\n')
    data_list = []
    for entry in entries:
        if not entry.strip():  # Skip empty lines
            continue
        data_dict = {
            "Date": "",
            "Details": "",
            "Parcels": "",
            "Net Amount": "",
            "Vat Amount": ""
        }
        parts = entry.split(', ')
        for part in parts:
            key_value = part.split(': ', 1)
            if len(key_value) == 2:
                key, value = key_value
                if key in data_dict:
                    data_dict[key] = value
        if any(data_dict.values()):  # Only add non-empty entries
            data_list.append(data_dict)
    return data_list

def create_docs(user_pdf_list):
    df = pd.DataFrame({
        "Date": pd.Series(dtype="str"),
        "Details": pd.Series(dtype="str"),
        "Parcels": pd.Series(dtype="str"),
        "Net Amount": pd.Series(dtype="str"),
        "Vat Amount": pd.Series(dtype="str"),
    })

    for filename in user_pdf_list:
        print(f"Processing {filename}")

        raw_data = get_pdf_text(filename)
        llm_extracted_data = extracted_data(raw_data)
        
        print("Extracted data:", llm_extracted_data[:500] + "...")  # Print first 500 characters
        
        data_list = parse_extracted_text(llm_extracted_data)
        
        for data_dict in data_list:
            print(data_dict)
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

        print(f"Finished processing {filename}")

    print("All files processed. DataFrame head:")
    print(df.head())
    return df