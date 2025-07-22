from pprint import pprint
from dotenv import load_dotenv
import os, getpass
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

messages = [SystemMessage(content="Eres un experto en matematicas, responde solo con el resultado")]
messages.append(HumanMessage(content=f"2+2",name="Alberto"))
messages.append(AIMessage(content=f"4", name="Model"))
messages.append(HumanMessage(content=f"ahora sumale 6",name="Alberto"))

for m in messages:
    m.pretty_print()

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini")
#result = llm.invoke(messages)
#print(result)

#TOOLS
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

tool_call = llm_with_tools.invoke([HumanMessage(content=f"dime la capital de españa", name="Alberto")])
result = tool_call.tool_calls
#print(result)



from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

print("------------------------")
print("Our graph")
print("------------------------")
class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass
    
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 1-Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
# 2- Logic
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
# 3-Add
graph = builder.compile()

# View
with open("graficagraphchain01.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()

messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()