import os
from haystack import Pipeline, Document
from haystack.dataclasses import ChatMessage
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from dotenv import load_dotenv

load_dotenv()

# Set your MongoDB connection string and OpenAI API key
# os.environ["MONGO_CONNECTION_STRING"] = os.getenv("MONGO_CONNECTION_STRING2")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the MongoDB Atlas Document Store
document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="common_info",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
    mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING2")

)

# Initialize a simple in-memory conversation history store
conversation_history = []

# Initialize the pipeline
rag_pipeline = Pipeline()

# Add a text embedder to embed the user's query
rag_pipeline.add_component(
    name="text_embedder",
    instance=SentenceTransformersTextEmbedder()
)

# Add the retriever to find relevant documents from the document store
rag_pipeline.add_component(
    name="retriever",
    instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=5)
)

# Add a prompt builder with memory to format the query, retrieved documents, and conversation history
prompt_template = """
You are a helpful assistant. Use the following conversation history and retrieved documents to answer the user's question. 
If the documents or history do not contain the answer, say that you don't have enough information.

Conversation History:
{% for message in conversation_history %}
    {{ message.role }}: {{ message.content }}
{% endfor %}

Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:
"""

rag_pipeline.add_component(
    name="prompt_builder",
    instance=PromptBuilder(template=prompt_template)
)

# Add the LLM to generate the final response
rag_pipeline.add_component(
    name="llm",
    instance=OpenAIGenerator(
        model="llama3-70b-8192",
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1"
    )
)

# Connect the components in the correct order
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

def run_query(user_query, conversation_history=None):
    """
    Run a query through the RAG pipeline.
    
    Args:
        user_query: The user's question
        conversation_history: The history of the conversation
        
    Returns:
        str: The generated answer
    """
    # Add user query to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_user(user_query))

    # Run the pipeline with conversation history
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": user_query},
            "prompt_builder": {
                "query": user_query,
                "conversation_history": conversation_history or []
            }
        }
    )

    # Get the generated answer
    answer = response['llm']['replies'][0]

    # Add AI response to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_assistant(answer))

    # Limit conversation history to the last 10 messages to prevent excessive growth
    if conversation_history is not None and len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    # Check for follow-up questions
    if "number" in user_query.lower():
        # Extract the number from the query
        number_match = re.search(r'\b(\d+)\b', user_query)
        if number_match:
            number = number_match.group(1)
            # Check if the answer contains the requested number
            if f"{number}." in answer:
                return f"Here is the description for number {number}: {answer.split(f'{number}.')[1].split('.')[0].strip()}"
            else:
                return "I don't have enough information to describe that number."
    """
    Run a query through the RAG pipeline.
    
    Args:
        user_query: The user's question
        conversation_history: The history of the conversation
        
    Returns:
        str: The generated answer
    """
    # Add user query to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_user(user_query))

    # Run the pipeline with conversation history
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": user_query},
            "prompt_builder": {
                "query": user_query,
                "conversation_history": conversation_history or []
            }
        }
    )

    # Get the generated answer
    answer = response['llm']['replies'][0]

    # Add AI response to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_assistant(answer))

    # Limit conversation history to the last 10 messages to prevent excessive growth
    if conversation_history is not None and len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    # Check for follow-up questions
    if "number" in user_query.lower():
        # Extract the number from the query
        number_match = re.search(r'\b(\d+)\b', user_query)
        if number_match:
            number = number_match.group(1)
            # Check if the answer contains the requested number
            if f"{number}." in answer:
                return f"Here is the description for number {number}: {answer.split(f'{number}.')[1].split('.')[0].strip()}"
            else:
                return "I don't have enough information to describe that number."

    """
    Run a query through the RAG pipeline.
    
    Args:
        user_query: The user's question
        conversation_history: The history of the conversation
        
    Returns:
        str: The generated answer
    """
    # Add user query to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_user(user_query))

    # Run the pipeline with conversation history
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": user_query},
            "prompt_builder": {
                "query": user_query,
                "conversation_history": conversation_history or []
            }
        }
    )

    # Get the generated answer
    answer = response['llm']['replies'][0]

    # Add AI response to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_assistant(answer))

    # Limit conversation history to the last 10 messages to prevent excessive growth
    if conversation_history is not None and len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    # Check for follow-up questions
    if "number" in user_query.lower():
        # Extract the number from the query
        number_match = re.search(r'\b(\d+)\b', user_query)
        if number_match:
            number = number_match.group(1)
            # Check if the answer contains the requested number
            if f"{number}." in answer:
                return f"Here is the description for number {number}: {answer.split(f'{number}.')[1].split('.')[0].strip()}"
            else:
                return "I don't have enough information to describe that number."

    """
    Run a query through the RAG pipeline.
    
    Args:
        user_query: The user's question
        conversation_history: The history of the conversation
        
    Returns:
        str: The generated answer
    """
    # Add user query to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_user(user_query))

    # Run the pipeline with conversation history
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": user_query},
            "prompt_builder": {
                "query": user_query,
                "conversation_history": conversation_history or []
            }
        }
    )

    # Get the generated answer
    answer = response['llm']['replies'][0]

    # Add AI response to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_assistant(answer))

    # Limit conversation history to the last 10 messages to prevent excessive growth
    if conversation_history is not None and len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    # Check for follow-up questions
    if "number" in user_query.lower():
        # Extract the number from the query
        number_match = re.search(r'\b(\d+)\b', user_query)
        if number_match:
            number = number_match.group(1)
            # Check if the answer contains the requested number
            if f"{number}." in answer:
                return f"Here is the description for number {number}: {answer.split(f'{number}.')[1].split('.')[0].strip()}"
            else:
                return "I don't have enough information to describe that number."
    """
    Run a query through the RAG pipeline.
    
    Args:
        user_query: The user's question
        conversation_history: The history of the conversation
        
    Returns:
        str: The generated answer
    """
    # Add user query to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_user(user_query))

    # Run the pipeline with conversation history
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": user_query},
            "prompt_builder": {
                "query": user_query,
                "conversation_history": conversation_history or []
            }
        }
    )

    # Get the generated answer
    answer = response['llm']['replies'][0]

    # Add AI response to conversation history
    if conversation_history is not None:
        conversation_history.append(ChatMessage.from_assistant(answer))

    # Limit conversation history to the last 10 messages to prevent excessive growth
    if conversation_history is not None and len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    return answer
    """
    Run a query through the RAG pipeline.
    
    Args:
        user_query: The user's question
        
    Returns:
        str: The generated answer
    """
    # Add user query to conversation history
    conversation_history.append(ChatMessage.from_user(user_query))

    # Run the pipeline with conversation history
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": user_query},
            "prompt_builder": {
                "query": user_query,
                "conversation_history": conversation_history
            }
        }
    )

    # Get the generated answer
    answer = response['llm']['replies'][0]

    # Add AI response to conversation history
    conversation_history.append(ChatMessage.from_assistant(answer))

    # Limit conversation history to the last 10 messages to prevent excessive growth
    if len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    return answer

# Example usage
# if __name__ == "__main__":
#     # Example query
#     user_query = "How can I cancel my order?"
#     answer = run_query(user_query)
#     print("Generated Answer:", answer)

#     # Follow-up query to demonstrate memory
#     follow_up_query = "What are the refund policies for canceled orders?"
#     answer = run_query(follow_up_query)
#     print("Generated Answer:", answer)
