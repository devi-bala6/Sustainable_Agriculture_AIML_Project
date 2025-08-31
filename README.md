# Sustainable_Agriculture_AIML_Project

# Myco-Net: The AI Fungal Network Interpreter

## Project Description

In an era of climate change and resource scarcity, small-scale farmers in India face significant challenges. Traditional farming methods are reactive, often leading to substantial crop loss due to water stress, nutrient deficiencies, and pests. Our project, **Myco-Net**, introduces a groundbreaking solution by using AI to interpret the "wood wide web" — the natural communication network of fungi in the soil.

This system is designed to provide **pre-emptive, real-time diagnostics**, enabling farmers to take action *before* crop damage becomes visible.

---

## Innovative Features

* **AI-Driven Signal Interpretation:** A machine learning model, trained on our custom dataset, interprets the subtle electrical and chemical signals from a plant's symbiotic fungal network.
* **Predictive Diagnostics:** The AI provides a precise diagnosis of plant health, identifying issues like water stress or pest attacks before they escalate.
* **Low-Cost & Scalable:** The project is designed to be affordable for a 1-acre farm, using a simple biosensor and a smartphone as the primary interface.

---

## Datasets Used

We have leveraged a combination of publicly available and simulated datasets to build and validate our AI model:

* **Kaggle "Plant Health Data" Dataset:** Our primary source for a clean, labeled dataset containing `Electrochemical_Signal` and `Plant_Health_Status` for AI training.
* **Simulated & Self-Collected Data:** Our raw sensor readings from various plant vases used for initial testing and proof-of-concept.
* **Research Papers (PDFs):** These documents serve as the scientific blueprint, providing a theoretical foundation for the electrical signals we are analyzing.

---

## Project Structure

This repository is organized to clearly document the project's development.
/Myco-Net/
|-- data/
|   |-- raw/
|   |-- preprocessed/
|-- src/
|   |-- ai_model.py
|   |-- data_preprocessing.py
|-- documents/
|-- README.md
