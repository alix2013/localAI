from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from store import retriever

model = OllamaLLM(model="gemma3")

template = """
You are an expert in answering questions about a restaurant
Here are some relevant information: {desc}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    docs = retriever.invoke(question)
    print("Relevant Documents:")
    print(docs)
    print("-"*50)
    result = chain.invoke({"desc": docs, "question": question})
    print(result)

