# Shop Recommendation System
This repository contains a shop recommendation system built with Python, leveraging Haystack for RAG (Retrieval-Augmented Generation) pipelines, MongoDB Atlas for document storage, and Streamlit for the web interface.
## Project Structure
```
├── Readme.md
├── data/
│   ├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
│   └── datasets.pkl
├── process/
│   ├── chat_memory.ipynb
│   ├── common_info_retrivier.ipynb
│   ├── common_info_store_data.ipynb
│   ├── generator.ipynb
│   ├── generator_filter.ipynb
│   ├── retriever.ipynb
│   ├── shop_recommendation.ipynb
│   └── store_data.ipynb
├── requirements.txt
└── website/
    ├── common_info.py
    ├── template.py
    └── website.py
```

## Main Components

1. data/: Contains datasets used for training and evaluation. <br>
2. process/: Jupyter notebooks for data processing, pipeline creation, and experimentation. <br>
3. common_info_retrivier.ipynb: RAG pipeline for general information retrieval.<br>
4. shop_recommendation.ipynb: Pipeline for shop/product recommendations.<br>
5. generator.ipynb, retriever.ipynb: LLM and retriever setup.<br>
6. website/: Streamlit web applications integrating Haystack pipelines for user interaction.<br>
7. requirements.txt: Python dependencies for the project.<br>


## Features

1. Shop/Product Recommendation: Uses RAG pipeline to recommend products based on user queries.<br>
2. General Information Retrieval: Answers general questions using a dedicated pipeline and MongoDB Atlas document store.<br>
3. Paraphrasing & Chat History: Includes paraphrasing and chat memory components for improved conversational experience.<br>
4. Streamlit Web Interface: User-friendly web app for interacting with the recommendation and information retrieval system.<br>

## Setup & Usage
1. Install Dependencies
```
pip install -r requirements.txt
```
2. Configure Environment Variables
Add your MongoDB Atlas connection string and GROQ API keys to process/.env and website/.env. I had to create two connection for MongoDB Atlas because limit of free tier. One for common information and the other else for shopping recommendation
3. Run the Streamlit App
```
streamlit run website/website.py
```

