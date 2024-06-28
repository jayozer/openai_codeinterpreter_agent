# PythonREPL Agent
# CSV Agent

from dotenv import load_dotenv

load_dotenv()

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain.chat_models import init_chat_model
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent


def main():
    print("Start...")

    python_agent = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tool=PythonREPLTool(),
        AgentType=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        verbose=True,
    )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True,
    )

    csv_agent.invoke(
        input={"input": "Which writer wrote the most episodes? How many episodes did he write?"}
    )


if __name__ == "__main__":
    main()
