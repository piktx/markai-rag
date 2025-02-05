import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe

# Set up Streamlit page
st.set_page_config(
    page_title="Smart Data Analysis with Groq API and PandasAI",
    layout="wide",
)

# Add custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
    }
    .css-1bc7jzt {
        color: #0072C6;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        color: #333333;
    }
    .stButton button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a powerful introductory prompt
st.markdown(
    """
    <div style='background-color: #0072C6; padding: 20px; border-radius: 10px; color: #FFFFFF; text-align: center;'>
        <h2>Welcome to Smart Data Analysis Platform</h2>
        <p>I am an experienced Python expert and Data Scientist here to assist you in analyzing and visualizing your data effortlessly. Upload your CSV or Excel file, ask questions in plain English, and let the power of AI provide insights!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Function to authenticate Groq API
@st.cache_resource
def authenticate_groq(api_key):
    return ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)

# Sidebar for Groq API authentication
st.sidebar.title("Authentication")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
authenticated = False

if groq_api_key:
    try:
        # Authenticate with the provided key
        groq_llm = authenticate_groq(groq_api_key)
        authenticated = True
        st.sidebar.success("Successfully authenticated!")
    except Exception as e:
        st.sidebar.error(f"Authentication failed: {e}")

if not authenticated:
    st.warning("Please authenticate with your Groq API Key in the sidebar to use the application.")
else:
    # Sidebar for file upload
    st.sidebar.title("Upload File")
    file_type = st.sidebar.radio("Select file type", ("CSV", "Excel"))
    uploaded_file = st.sidebar.file_uploader(f"Upload a {file_type} file", type=["csv", "xls", "xlsx"])

    # Main application content
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if file_type == "CSV":
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file, engine="openpyxl")

            # Display data preview
            st.subheader("Data Preview")
            st.write(data.head())

            # Display general information about the data
            st.subheader("General Information")
            st.write(f"Shape of the dataset: {data.shape}")
            st.write(f"Data Types:\n{data.dtypes}")
            st.write(f"Memory Usage: {data.memory_usage(deep=True).sum()} bytes")

            # SmartDataframe setup for AI-driven interaction
            df_groq = SmartDataframe(data, config={'llm': groq_llm})

            # User input for natural language query
            query = st.text_input(
                "Ask me anything about your data, and I'll provide clear insights or visualizations:",
                placeholder="For example, 'Show me the average sales per month' or 'Visualize the top 5 products by revenue.'"
            )

            if query:
                try:
                    # Query processing
                    start_time = time.time()
                    response = df_groq.chat(query)
                    end_time = time.time()

                    # Dynamically render charts based on query
                    st.subheader("Generated Visualization")

                    if "distribution" in query.lower():
                        column = "popularity"  # Default column for distribution
                        for col in data.columns:
                            if col.lower() in query.lower():
                                column = col
                                break
                        # Plotly interactive visualization
                        fig = px.histogram(data, x=column, nbins=30, title=f"Distribution of {column.capitalize()}")
                        fig.update_layout(xaxis_title=column.capitalize(), yaxis_title="Frequency")
                        st.plotly_chart(fig)
                    elif "scatter" in query.lower() and "vs" in query.lower():
                        cols = query.lower().split("vs")
                        col_x = cols[0].split()[-1].strip()
                        col_y = cols[1].strip()
                        if col_x in data.columns and col_y in data.columns:
                            # Plotly scatter plot
                            fig = px.scatter(data, x=col_x, y=col_y, title=f"Scatter Plot of {col_x.capitalize()} vs {col_y.capitalize()}")
                            fig.update_layout(xaxis_title=col_x.capitalize(), yaxis_title=col_y.capitalize())
                            st.plotly_chart(fig)
                        else:
                            st.error("Columns specified for scatter plot not found in the dataset.")
                    elif "bar" in query.lower():
                        column = "artist_name"  # Example: Use an artist_name column
                        if column in data.columns:
                            fig = px.bar(data, x=column, y="popularity", title="Bar Chart Example")
                            st.plotly_chart(fig)
                        else:
                            st.error("The specified column for the bar chart does not exist.")
                    else:
                        st.write(response)
                    st.success(f"Query processed in {end_time - start_time:.2f} seconds.")
                except Exception as e:
                    st.error(f"An error occurred during query processing: {e}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Footer
    st.markdown(
        """
        <footer style='text-align: center; padding: 10px; background-color: #0072C6; color: #FFFFFF;'>
            Powered by AI and Open Source Tools<br>
            Made with ❤️ by Martin Khristi
        </footer>
        """,
        unsafe_allow_html=True,
    )
