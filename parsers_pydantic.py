import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4"
llm = ChatOpenAI(temperature=0.7, model =llm_model)

email_response = """Here is our plan, we will leave for Pakistan at 9am ,and will visit karachi and 
Lahore, there will be 6 of us on the trip"""

email_template = """extract this

leave_time

cities_to_viist

format hte output as JSON with the following keys
leave_time
cities_to_visit

email: {email}
"""


from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

class Vacation(BaseModel):
    leave_time: str = Field(description="When they are leaving, what time")
    cities_to_visit: List = Field(description= "Which cities are we visiting")
    num_people: int = Field(description="this is an int of the number of people on the trip")
    
    @validator('num_people')
    def check_num_people(cls, field):
        if field <= 0:
            raise ValueError("badly formatted field")
        return field
    
pydantic_parser = PydanticOutputParser(pydantic_object=Vacation)
format_ins = pydantic_parser.get_format_instructions()
    

email_template_revised = """
From the following email, extract the following information regarding this trip

email: {email}

{format_ins}
"""

updated_prompt = PromptTemplate.from_template(template=email_template_revised)
messages = updated_prompt.format(email=email_response,
                                 format_ins=format_ins)

format_response = llm(messages)

# print(format_response.content)

vacation = pydantic_parser.parse(format_response.content)
print(vacation.cities_to_visit)