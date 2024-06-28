# PythonREPL Agent
# CSV Agent
# Router Agent
# OpenAI Functions
# langchain-experiemental - used for prerelease.
# qrcode is installed for testing
# https://smith.langchain.com/hub/langchain-ai/react-agent-template

from dotenv import load_dotenv

load_dotenv()
from langchain import hub

# experimental - instead of directly instantiating the ChatOpenAI class, init_chat_model() helper function checks the availability and access requirements for the model.
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool


def main():
    print("Start...")
    # addtional instructions to be added to react prompt
    instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        You have qrcode package installed
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=init_chat_model(
            temperature=0, model="gpt-4o"
        ),  # using init_chat_model for gpt-4o
        tools=tools,
    )

    python_agent_executor = AgentExecutor(
        agent=python_agent, tools=tools, verbose=True
    )  # verbose = True is to see extra logs

    # invoke agent
    python_agent_executor.invoke(
        input={
            "input": """generate and save in current working directory in a folder called qrcode2 15 QRcodes
            that point to www.poppykidsdental.com, you have qrcode package installed already."""
        }
    )


if __name__ == "__main__":
    main()
