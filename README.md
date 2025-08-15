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

data/: Contains datasets used for training and evaluation. <br>
process/: Jupyter notebooks for data processing, pipeline creation, and experimentation. <br>
common_info_retrivier.ipynb: RAG pipeline for general information retrieval.<br>
shop_recommendation.ipynb: Pipeline for shop/product recommendations.<br>
generator.ipynb, retriever.ipynb: LLM and retriever setup.<br>
website/: Streamlit web applications integrating Haystack pipelines for user interaction.<br>
requirements.txt: Python dependencies for the project.<br>
tf_env/: Python virtual environment (do not edit directly).<br>


## Features

Shop/Product Recommendation: Uses RAG pipeline to recommend products based on user queries.<br>
General Information Retrieval: Answers general questions using a dedicated pipeline and MongoDB Atlas document store.<br>
Paraphrasing & Chat History: Includes paraphrasing and chat memory components for improved conversational experience.<br>
Streamlit Web Interface: User-friendly web app for interacting with the recommendation and information retrieval system.<br>

## Setup & Usage

