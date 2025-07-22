from dotenv import load_dotenv
from IPython.display import Image, display
import operator
from typing import Annotated

from typing import Any
from typing_extensions import TypedDict
from langgraph.errors import InvalidUpdateError

from langgraph.graph import StateGraph, START, END

load_dotenv()

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    state: str

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

#1 # Sequential example
# This example shows how to create a state graph with sequential execution.
# It uses a simple state graph with nodes that return a value.
# The nodes are executed in a sequence, and the final state is printed.

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

# View
with open("parallelization01.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

print(graph.invoke({"state": []}))


#2 # Parallelization example
# This example shows how to create a state graph with parallelization.
# It uses the same nodes as the previous example, but allows for parallel execution of nodes.
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

with open("parallelization02.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("--------------------------")
print("Parallelized graph:")
try:
    print(graph.invoke({"state": []}))
except InvalidUpdateError as e:
    print(f"An error occurred: {e}")


class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    state: Annotated[list, operator.add]

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

with open("parallelization03.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

print("--------------------------")
print("Parallelized graph: operator.add")
print("--------------------------")
print(graph.invoke({"state": []}))

