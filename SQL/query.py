# code to generate query

import os
import deepspeed
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments



quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    offload_folder="weights",
    load_in_8bit=True,
    torch_dtype = torch.float16,
    device_map='auto',
    quantization_config = quantization_config,
    
)
tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)


template = """
Write a Postgres SQL Query given the table name {Table} and columns as a list {Columns} for the given question : 
{question}.
"""

prompt = PromptTemplate(template=template, input_variables=["Table","question","Columns"])
llm_chain = LLMChain(prompt=prompt, llm=local_llm)
def get_llm_response(tble,question,cols):
    llm_chain = LLMChain(prompt=prompt, 
                         llm=local_llm
                         )
    response= llm_chain.run({"Table" : tble,"question" :question, "Columns" : cols})
    print(response)

tble = "employee"
cols = ["id","name","date_of_birth","band","manager_id"]
question = "Query the count of employees in band L6 with 239045 as the manager ID"
get_llm_response(tble,question,cols)