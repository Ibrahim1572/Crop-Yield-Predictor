import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import altair as alt
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "crop_yield_short.csv"
image_pathh = BASE_DIR / "background2.jpg"
text_path = BASE_DIR / "crop_yield_shorter.txt"

st.set_page_config(page_title="Yield Wizzard 3000", layout="centered")

def set_bg_with_overlay(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, 0.7), 
            rgba(0, 0, 0, 0.7)
        ), url("data:image/jpg;base64,{encoded}"); 
        background-size: cover;
        background-position: center;
    }}

    .title {{
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin: 20px 0;
        color: white;
    }}
    
    /* Centered tab container */
    div[data-testid="stTabs"] {{
        display: flex;
        justify-content: center;
    }}
    
    /* Centered tab list */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        padding: 10px 0;
        display: flex;
        justify-content: center;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        border-radius: 20px !important;
        background-color: white !important;
        color: #333 !important;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: #f0f0f0 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: #4a8fe7 !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(74, 143, 231, 0.3);
    }}
    
    /* Remove the bottom border line */
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: transparent !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_with_overlay(image_pathh)

df_plots = pd.read_csv(csv_path)
df_model = pd.read_csv(csv_path)

if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

# Loop until valid API key is entered
if not st.session_state.api_key_valid:
    st.sidebar.header("🔐 API Key Setup")
    user_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

    if user_api_key:
        try:
            genai.configure(api_key=user_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            _ = model.generate_content("Test")  # Dummy request

            # If successful
            st.session_state.api_key_valid = True
            st.success("API key is valid!")

            # Hide sidebar after success
            hide_sidebar = """
            <style>
                [data-testid="stSidebar"] {
                    display: none;
                }
                [data-testid="collapsedControl"] {
                    display: none;
                }
            </style>
            """
            st.markdown(hide_sidebar, unsafe_allow_html=True)

        except Exception:
            st.sidebar.error("Invalid API key. Please enter a valid key.")
            st.stop()
    else:
        st.warning("Please enter your Gemini API key to proceed.")
        st.stop()

graph_type = ['Bar Graph', 'Scatter plot', 'Box Plot', 'Line Chart']

def make_graph(typ):
    if typ == graph_type[3]:
        grouped = df_plots.groupby(['Crop', 'Region']).size().reset_index(name='Frequency')
        color_scale = alt.Scale(domain=["North", "South", "East", "West"], range=["#1f77b4", "#2ca02c", "#d62728", "#ffc107"])
        chart = alt.Chart(grouped).mark_line(point=True).encode(
            x='Crop:N',
            y='Frequency:Q',
            color=alt.Color('Region:N', scale=color_scale, legend=alt.Legend(title="Region"))
        ).properties(width=700, height=400, title="Crop Frequency by Region (Custom Colors)")
        st.altair_chart(chart, use_container_width=True)
    elif typ == graph_type[0]:
        avg_yield = df_plots.groupby("Crop")["Yield_tons_per_hectare"].mean().reset_index()
        colors = ['#FFFF00', '#FFD700', '#FFA500', '#FF7F00', '#FF4500', '#FF0000']
        unique_crops = avg_yield['Crop'].unique()
        color_map = {crop: colors[i % len(colors)] for i, crop in enumerate(unique_crops)}
        avg_yield['Color'] = avg_yield['Crop'].map(color_map)
        chart = alt.Chart(avg_yield).mark_bar().encode(
            x=alt.X("Crop:N", title="Crop"),
            y=alt.Y("Yield_tons_per_hectare:Q", title="Average Yield (Ton per Hectare)"),
            color=alt.Color("Crop:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=None)
        ).properties(title="Average Yield per Crop", width=100, height=400)
        st.altair_chart(chart, use_container_width=True)
    elif typ == graph_type[1]:
        scatter = alt.Chart(df_plots).mark_circle(size=100).encode(
            x=alt.X("Rainfall_mm", title="Rainfall (mm)"),
            y=alt.Y("Yield_tons_per_hectare", title="Yield (tons/hectare)"),
            color=alt.Color("Crop", legend=alt.Legend(title="Crop")),
            tooltip=["Crop", "Rainfall_mm", "Yield_tons_per_hectare"]
        ).properties(title="Yield vs. Rainfall Scatter Plot", width=600, height=400).interactive()
        st.altair_chart(scatter, use_container_width=True)
    elif typ == graph_type[2]:
        color_scale = alt.Scale(domain=["Cloudy", "Rainy", "Sunny"], range=["#a6cee3", "#e31a1c", "#1b9e77"])
        chart = alt.Chart(df_plots).mark_boxplot(extent='min-max').encode(
            x=alt.X('Weather_Condition:N', title='Weather Condition'),
            y=alt.Y('Yield_tons_per_hectare:Q', title='Yield (tons/hectare)'),
            color=alt.Color('Weather_Condition:N', scale=color_scale, legend=alt.Legend(title="Weather"))
        ).properties(width=1000, height=400, title="Yield vs. Weather Condition (Box Plot)")
        st.altair_chart(chart, use_container_width=True)

# Set the background image and title
st.markdown(f'<div class="title">Yield Wizzard 3000</div>', unsafe_allow_html=True)

# Create centered tabs
col1, col2, col3 = st.columns([1,2,1])
with col2:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Frame", "Graph Options", "Summary and Stats", "Model Training", "About developers"])

    with tab1:
        st.subheader("Data")
        st.dataframe(df_plots)

    with tab2:
        st.subheader('Graph options')
        choice_graph_type = st.selectbox("Choose Graph Type", list(graph_type))
        make_graph(choice_graph_type)


    with tab3:
        st.subheader('Get Statistical analysis with the Chat Wizzard')

        with st.form("analysis_form"):
            st.subheader("\U0001F4C8 Select Graph for Analysis")
            graph_target = st.selectbox(
                "Which relationship or comparison do you want to analyze?",
                [
                    "Crop vs Region",
                    "Rainfall vs Yield",
                    "Temperature vs Yield",
                    "Soil Type vs Yield",
                    "Weather Condition vs Crop",
                ]
            )

            st.subheader("\U0001F4CA Chart Types Used")
            chart_bar = st.checkbox("Bar Chart")
            chart_line = st.checkbox("Line Chart")
            chart_box = st.checkbox("Box Plot")
            chart_scatter = st.checkbox("Scatter Plot")

            st.subheader("❓ Custom Questions for Analysis")
            question_1 = st.text_input("Question 1", placeholder="e.g., Which region gives highest yield for Cotton?")
            question_2 = st.text_input("Question 2", placeholder="e.g., Is higher rainfall always better for crops?")
            question_3 = st.text_input("Question 3", placeholder="e.g., Do Peaty soils outperform Loam in wheat growth?")

            st.subheader("\U0001F529 Optional Data Filters")
            temp_range = st.slider("Temperature Range (°C)", 10, 45, (15, 35))
            rainfall_range = st.slider("Rainfall Range (mm)", 0, 1200, (100, 1000))

            submitted = st.form_submit_button("Ask Wizzi")

            if submitted:
                chart_types = []
                if chart_bar: chart_types.append("Bar Chart")
                if chart_line: chart_types.append("Line Chart")
                if chart_box: chart_types.append("Box Plot")
                if chart_scatter: chart_types.append("Scatter Plot")
                charts_used = ", ".join(chart_types) if chart_types else "No charts selected"

                final_prompt = f"""
    You are a data analyst. You are given a dataset and a visualization showing {graph_target}. Analyze the data using statistical reasoning.

    Charts Used:
    {charts_used}

    Analysis Tasks:
    1. Identify patterns and trends in the relationship between {graph_target}.
    2. Detect any outliers or unusual values.
    3. Comment on potential correlations or regional/categorical performance.
    4. Answer these user questions:
    - {question_1 or '[User Question 1]'}
    - {question_2 or '[User Question 2]'}
    - {question_3 or '[User Question 3]'}

    User-specified Filters:
    - Temperature Range: {temp_range[0]}°C to {temp_range[1]}°C
    - Rainfall Range: {rainfall_range[0]} mm to {rainfall_range[1]} mm

    Instructions:
    Respond in paragraphs or bullet points. Use basic statistical terms (mean, range, correlation) where applicable.
    Use a maximum of 2 paragraphs with up to 5 lines each. Each paragraph can replace 5 bullet points and vice versa. at the end answer the user asked question seperatly. make the paragraphs consistent with 5 lines max.
    """

                with open(text_path, "r") as f:
                    file_text = f.read()

                combined_prompt = final_prompt + "\n\n\U0001F4C4 Dataset Snapshot:\n" + file_text

                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(combined_prompt)

                st.subheader("\U0001F4CB AI Analysis Output")
                st.write(response.text)

df_model['Region']= df_model['Region'].replace({'North':1.0, 'South':2.0, 'East':3.0, 'West':4.0})
df_model['Soil_Type']= df_model['Soil_Type'].replace({'Sandy':1.0, 'Clay':2.0, 'Loam':3.0, 'Silt':4.0, 'Peaty':5.0, 'Chalky':6.0})
df_model['Crop']=df_model['Crop'].replace({'Cotton':1.0, 'Rice':2.0, 'Barley':3.0, 'Soybean':4.0, 'Wheat':5.0, 'Maize':6.0})
df_model['Irrigation_Used']=df_model['Irrigation_Used'].replace({True:1.0, False:0.0})
df_model['Fertilizer_Used']=df_model['Fertilizer_Used'].replace({True:1.0, False:0.0})
df_model['Weather_Condition']=df_model['Weather_Condition'].replace({'Cloudy':1.0, 'Rainy':2.0, 'Sunny':3.0})
df_model['Days_to_Harvest']=df_model['Days_to_Harvest'].astype(float)

x=df_model[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
       'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition',
       'Days_to_Harvest']]
y=df_model['Yield_tons_per_hectare']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred= model.predict(x_test)

MSE= mean_squared_error(y_test, y_pred)
R2= r2_score(y_test, y_pred)

    
with tab4:
    st.subheader("🧠 Learning Model Results")
    st.write("**Learning Model used:** Linear Regression")
    st.write(f"**MSE:** {MSE}")
    st.write(f"**R² Score:** {R2}")
    
with tab5:
    st.subheader("👨‍💻 About the Developers")
    st.markdown("""
        ### Yield Wizzard 3000
        Developed as a part of the Artificial Intelligence Lab Project at GIK Institute.

        **Team Members:**
        - **Muhammad Ibrahim Umar** (2024413) 
        - **Muhammad Shahnawaz**    (2024464)

        **Tools & Technologies:**
        - Python, Pandas, NumPy, Streamlit
        - Machine Learning (Linear Regression)
        - Altair, Plotly, Seaborn
        - Gemini API for LLM-driven analysis

        **Project Objective:**
        To visualize and predict agricultural crop yield based on environmental and regional data.
        """, unsafe_allow_html=True)