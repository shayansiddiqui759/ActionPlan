
import openai

from helper import get_env, file_helper, agent_get_func_helper
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents import create_react_agent

from langchain.memory import ConversationBufferMemory

from model import param



from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

# GET API KEY
DEEPSEEK_API_KEY= get_env.retreive_value( "DEEPSEEK_API_KEY")
MISTRAL_API_KEY = get_env.retreive_value( "MISTRAL_API_KEY")
OPENAI_API_KEY  = get_env.retreive_value( "OPENAI_API_KEY")
openai.api_key  = OPENAI_API_KEY

AGENT_MODEL     =  get_env.retreive_value( "AGENT_MODEL_MISTRAL")

ACTIONPLAN_VECTOR = "vector_db"


 
POPERTY_PATH = get_env.retreive_value( "PROPERTY_PATH")
POPERTY = file_helper.read_json( POPERTY_PATH ) 

template_str     = POPERTY.get("template_str")
tool_description = POPERTY.get("tool_description")

def agent_executor( question: str ) -> str:
    
    # GET CHAT MODEL
    chat_model = agent_get_func_helper.get_chat_model("misrtal") 
    
    
    # SYSTEM
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=template_str,
        )
    )

    # HUMAN
    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )

    # COMBIME MESSAGE
    messages = [system_prompt, human_prompt]

    # SET UP PROMPT TEMPLATE
    prompt_template = ChatPromptTemplate( input_variables=["context", "question"], 
                                          messages=messages, )
    vectordb = Chroma(
            persist_directory=ACTIONPLAN_VECTOR,
            embedding_function=OpenAIEmbeddings(),
    )

    retriever = vectordb.as_retriever(search_type="similarity", 
                                      search_kwargs={"k": 20})

    vector_chain = (
            {"context": retriever, "question":  RunnablePassthrough() }
            | prompt_template
            | chat_model
            | StrOutputParser()
    )

    tools = [
        Tool(
            name="ActionPlan",
            func=vector_chain.invoke,
            description= tool_description,
        ),
      
    ]

    # agent_prompt = hub.pull("hwchase17/openai-functions-agent") # DOWNLOAD PREBUILD PROMPT
    agent_prompt = hub.pull("hwchase17/react-chat") # Use react chat prompt that works with any llm.

    # Create Agent
    agent = create_react_agent(chat_model, tools, agent_prompt)



    # Create Agent Executor
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # memory to remember past conversations
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = memory)

    # # Example Usage
    question = "I am 30 years old and have a BRCA2 gene fault. What are the risks of developing cancer?"
    response = agent_executor.invoke({"input": question})

    return response



#     ####### Prompt Template #######
#     # Instructions:
#     # Use the provided context to answer questions.
#     # Be detailed and include quantitative information.
#     # Reference specific sections of the context (e.g., "Heading: Resources").
#     # Handle invalid or vague questions gracefully.
#     # Avoid stating "I am an AI assistant."
#     # Defer to a genetic counselor if the answer is unavailable.

#     ####### HOW CODE WORKS #######
#     # Input: A user question about cancer risks.
#     # Process:
#     # The question is passed to the agent.
#     # The agent retrieves relevant context, combines it with the question, and generates a response.
#     # Output: The response is printed.

#     # Summary
#     # Tools: Define what the agent can do (e.g., retrieve data).
#     # Prompt: Guides the agent's behavior.
#     # Memory: Helps the agent remember past conversations.
#     # Executor: Runs the agent and handles user interactions.