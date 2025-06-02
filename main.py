from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver  # Import the retriever from the vector module

model = OllamaLLM(model="llama3.2") # This creates an instance of the OllamaLLM class
template = """
You are an expert in answering questions about the pizza restaurant.
Here are some relevant reviews:{reviews}
Here are the question to answer:{question}
"""
prompt = ChatPromptTemplate.from_template(template) # This creates a prompt template from the provided string

chain = prompt | model # This creates a chain that combines the prompt and the model

while True:
    print("\n\n------------------------------------")
    question = input("Enter your question (q to quit): ") # This prompts the user for a question
    print("\n\n------------------------------------")
    if question == 'q': # If the user enters 'q', the loop breaks
        break

    reviews = retriver.invoke(question) # This retrieves relevant reviews based on the user's question
    result = chain.invoke({
    "reviews": reviews,
    "question": question
    }) # This invokes the chain with the specified inputs
    print(result) # This prints the result of the chain invocation