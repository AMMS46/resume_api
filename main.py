from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_google_genai import GoogleGenerativeAI
import io
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

resume_template = """
You are bot which takes data and generate ATS friendly resume in the following form,

Name:
contact:
Mail:
Github:
LinkedIn:


Profile:
4 to  5 lines max

Education:
Top 2 from the given data

Work/Professional experience:
Top 3  from the given data

Projects:
Top 2 from the given data

Skills:
Separated by comas


Question: {question}
Answer:"""



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeData(BaseModel):
    data: str


@app.post("/data")
async def get_data(resume_data: ResumeData):
    
    llm= GoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY,max_tokens = 256)
    
    code_assistant_prompt_template = PromptTemplate(
        input_variables=["data"],
        template=resume_template)
    
    llm_chain = LLMChain(llm=llm,
                         prompt=code_assistant_prompt_template)
    
    response = llm_chain.run({"data": resume_data.data})
    return {"response": response}

    
     
   