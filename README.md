Sustainable_Agriculture_AIML_Project
🌿 Myco-Net: The AI Fungal Network Interpreter
Myco-Net is a groundbreaking AI solution designed to transform sustainable agriculture from a reactive to a proactive process. By interpreting subtle signals from the “wood wide web”—the natural communication network of fungi in the soil—our system delivers real-time diagnostics and early warnings of plant stress, enabling farmers to take action before crop damage becomes visible.

🚀 Key Innovations
AI-Driven Signal Interpretation
A sophisticated machine learning pipeline analyzes complex biological and environmental data, uncovering hidden patterns that indicate plant stress.

Proactive Diagnostics
Unlike traditional approaches that rely on visible symptoms, Myco-Net provides pre-emptive detection of water stress, nutrient deficiencies, and soil-borne pathogens. 🌾

Two-Tiered Validation
Our system validates predictions across two phases:

A foundational dataset of general plant health metrics.

A specialized dataset enriched with fungal communication signals.

📂 Project Structure
bash
Copy code
/
├── data/
│   ├── plant_health_data.csv        # Dataset for foundational analysis
│   └── Tree_Data.csv                # Dataset for core Myco-Net analysis
├── src/
│   ├── app.py                       # Streamlit web application
│   └── myco_net_analysis.ipynb      # Jupyter notebook with analysis & training
├── models/
│   ├── myco_net_model.pkl           # Trained Myco-Net model
│   ├── plant_health_model.pkl       # Trained plant health model
│   ├── species_encoder.pkl          # Label encoder for plant species
│   ├── light_encoder.pkl            # Label encoder for light conditions
│   └── microbe_encoder.pkl          # Label encoder for microbial communities
└── README.md                        # Project documentation
⚙️ How to Run the Project
1. Clone the Repository
bash
Copy code
git clone https://github.com/[Your_Username]/[Your_Repo_Name].git
cd [Your_Repo_Name]
2. Run the Analysis Notebook
Navigate to the src/ directory.

Open myco_net_analysis.ipynb in Jupyter Notebook or JupyterLab.

Run all cells from top to bottom to explore preprocessing, feature engineering, and model training.

3. Launch the Web Application
Ensure dependencies are installed:

bash
Copy code
pip install streamlit pandas scikit-learn matplotlib seaborn
Run the app:

bash
Copy code
streamlit run src/app.py
This will open the interactive Myco-Net dashboard in your web browser.

📖 About
Myco-Net: The AI Fungal Network Interpreter
Empowering small-scale farmers with AI-powered insights from fungal networks, making agriculture more resilient, proactive, and sustainable. 🌱
