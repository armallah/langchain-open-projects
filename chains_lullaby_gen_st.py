import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st
from langchain_openai import OpenAI



load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4"

llm = OpenAI(temperature=0.7)

def generate_lullaby(location, name, langauge):
    
    template = """
    As a children's book writer, please come up with a short and sweet (90 words) poem based on the location
    {location}
    and the main character {name}

    STORY:
    """

    prompt = PromptTemplate(input_variables=["location", "name"],
                            template = template)
    
    from langchain_core.output_parsers import StrOutputParser
    

    chain_story = prompt | llm | StrOutputParser()

    ##sequential chain

    open_ai = OpenAI(temperature=0.7)

    template = """
    As a children's book writer, please come up with a short and sweet (90 words) poem based on the location
    {location}
    and the main character {name}

    STORY:
    """

    prompt = PromptTemplate(input_variables=["location", "name"],
                            template = template)

    # chain_story = prompt | llm | StrOutputParser()
    chain_story = LLMChain(llm=open_ai, prompt=prompt, output_key="story")

    story = chain_story.invoke({"location" :"Zanibar", "name": "Maya"})

    from langchain.chains import SequentialChain

    template_update = """
    Translate the {story} into {language}, make sure the language is simple

    TRANSLATION:"""

    prompt_translate = PromptTemplate(input_variables=["story","language"],
                                    template= template_update)

    # chain_trans = prompt_translate | llm | StrOutputParser()

    chain_trans = LLMChain(llm=open_ai, prompt=prompt_translate, output_key = "translated")
    overall_chain = SequentialChain(
        chains = [chain_story, chain_trans],
        input_variables = ["location", "name", "language"],
        output_variables = ["story", "translated"]
    )

    response = overall_chain({"location":location,
                            "name":name,
                            "language": langauge})


    return response


def main():
    
    st.set_page_config(page_title="Generate Children's Story", layout= "centered")
    st.title("Let AI Write and Translate a Story for You 📖")
    st.header("Get Started...")
    # pass
    location_input = st.text_input(label="Where is the story set?")
    character_input = st.text_input(label="What's the main character called?")
    language_input = st.text_input(label="Translate the story into...")
    

    submit_button = st.button("Submit")
    
    if location_input and character_input and language_input:
        if submit_button:
            with st.spinner("Generating story"):
            
                response = generate_lullaby(location=location_input, 
                                            name = character_input,
                                            langauge=language_input)
                
                with st.expander("English version"):
                    st.write(response['story'])
                with st.expander(f"{language_input} Version"):
                    st.write(response['translated'])
                    
            st.success("Story generated")


if __name__ == "__main__":
    main()