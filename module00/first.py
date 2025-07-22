from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

modelChatOpenAI = ChatOpenAI(model="gpt-4o-mini")
modelInitChatModel = init_chat_model(model_provider="openai", model="gpt-4o-mini")

messages = [
    SystemMessage(content="Eres un profesor de matemáticas. Y explicas las cosas de forma sencilla."),
    HumanMessage(content="2+2"),
]

result = modelChatOpenAI.invoke(messages)
print(f"Answer from AI: {result.content}")
print(f"-------------------------------\n")
result = modelInitChatModel.invoke(messages)
print(f"Answer from AI: {result.content}")