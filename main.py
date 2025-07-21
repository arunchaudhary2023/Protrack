import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time
from PIL import Image
import cv2
import os
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Project Progress Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stDateInput>div>div>input {
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .status-completed {
        color: #28a745;
        font-weight: bold;
    }
    .status-new {
        color: #17a2b8;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'progress_data' not in st.session_state:
    st.session_state.progress_data = pd.DataFrame(columns=['Project', 'Date', 'Progress', 'Notes'])
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = {}


# Mock ML model for delay prediction
def predict_delay(project_data):
    # This is a simplified mock model
    progress_rate = project_data['Progress'].pct_change().mean()
    if pd.isna(progress_rate) or progress_rate <= 0:
        return "High risk (estimated 2+ weeks delay)"

    target_progress = 100
    current_progress = project_data['Progress'].iloc[-1]
    days_elapsed = (project_data['Date'].iloc[-1] - project_data['Date'].iloc[0]).days

    if progress_rate > 0:
        days_remaining = (target_progress - current_progress) / progress_rate
        planned_days = days_elapsed * (target_progress / current_progress - 1)
        delay_days = days_remaining - planned_days

        if delay_days < 3:
            return "On schedule"
        elif 3 <= delay_days < 7:
            return "Slight delay (estimated 3-7 days)"
        elif 7 <= delay_days < 14:
            return "Moderate delay (estimated 1-2 weeks)"
        else:
            return "High risk (estimated 2+ weeks delay)"
    return "Unable to predict"


# Mock image analysis function
def analyze_image_progress(image, project_name):
    # In a real app, this would use computer vision models
    # For demo purposes, we'll return a random progress between last progress and 100%
    if project_name in st.session_state.progress_data['Project'].values:
        last_progress = st.session_state.progress_data[
            st.session_state.progress_data['Project'] == project_name
            ]['Progress'].max()
        progress = min(100, last_progress + np.random.randint(5, 20))
    else:
        progress = np.random.randint(10, 30)

    # Save uploaded image for display
    if project_name not in st.session_state.uploaded_images:
        st.session_state.uploaded_images[project_name] = []
    st.session_state.uploaded_images[project_name].append(image)

    return progress


# Sidebar - Project Management
with st.sidebar:
    st.title("📊 Project Management")

    # Project selection/create
    project_option = st.radio("Project Options", ["Select Project", "Create New Project"])

    if project_option == "Create New Project":
        with st.form("new_project_form"):
            project_name = st.text_input("Project Name")
            start_date = st.date_input("Start Date")
            target_date = st.date_input("Target Completion Date")
            target_progress = st.number_input("Target Progress (%)", min_value=1, max_value=100, value=100)
            submit_project = st.form_submit_button("Create Project")

            if submit_project and project_name:
                if project_name in st.session_state.projects:
                    st.error("Project name already exists!")
                else:
                    st.session_state.projects[project_name] = {
                        'start_date': start_date,
                        'target_date': target_date,
                        'target_progress': target_progress,
                        'created_at': datetime.now()
                    }
                    st.session_state.current_project = project_name
                    st.success(f"Project '{project_name}' created!")

    else:
        if st.session_state.projects:
            selected_project = st.selectbox(
                "Select Project",
                list(st.session_state.projects.keys()),
                index=list(st.session_state.projects.keys()).index(st.session_state.current_project)
                if st.session_state.current_project in st.session_state.projects else 0
            )
            st.session_state.current_project = selected_project
        else:
            st.info("No projects available. Create a new project.")

    # Display project info
    if st.session_state.current_project:
        st.subheader("Project Info")
        project = st.session_state.projects[st.session_state.current_project]
        st.write(f"**Start Date:** {project['start_date']}")
        st.write(f"**Target Date:** {project['target_date']}")
        st.write(f"**Target Progress:** {project['target_progress']}%")

        # Calculate days remaining
        days_remaining = (project['target_date'] - datetime.now().date()).days
        st.write(f"**Days Remaining:** {days_remaining}")

# Main content
st.title("📊 Project Progress Monitoring Dashboard")

if st.session_state.current_project:
    project_name = st.session_state.current_project
    project_data = st.session_state.projects[project_name]

    # Project metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Calculate current progress
        project_progress_data = st.session_state.progress_data[
            st.session_state.progress_data['Project'] == project_name
            ]
        current_progress = project_progress_data['Progress'].max() if not project_progress_data.empty else 0

        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Progress</h3>
            <h1>{current_progress}%</h1>
            <st.progress value="{current_progress}"></st.progress>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Days remaining
        days_remaining = (project_data['target_date'] - datetime.now().date()).days
        st.markdown(f"""
        <div class="metric-card">
            <h3>Days Remaining</h3>
            <h1>{days_remaining}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Progress needed per day
        progress_needed = project_data['target_progress'] - current_progress
        daily_progress_needed = progress_needed / days_remaining if days_remaining > 0 else progress_needed
        st.markdown(f"""
        <div class="metric-card">
            <h3>Daily Progress Needed</h3>
            <h1>{daily_progress_needed:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Delay prediction
        if not project_progress_data.empty:
            delay_prediction = predict_delay(project_progress_data)
            status_color = "green" if "On schedule" in delay_prediction else "orange" if "Slight" in delay_prediction else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Status</h3>
                <h1 style="color: {status_color}">{delay_prediction}</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Status</h3>
                <h1>No data yet</h1>
            </div>
            """, unsafe_allow_html=True)

    # Features section
    st.subheader("Progress Monitoring Features")

    # Feature cards
    tab1, tab2, tab3, tab4 = st.tabs([
        "Manual Progress Input",
        "Auto Progress Estimation",
        "Delay Prediction",
        "Progress Charts"
    ])

    with tab1:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="status-completed">✅</span> Manual Task Progress Input</h3>
            <p>Enter percentage of completion manually or via file upload</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Add Manual Progress Update", expanded=True):
            with st.form("manual_progress_form"):
                progress_date = st.date_input("Date", datetime.now().date())
                progress_value = st.slider("Progress (%)", 0, 100, current_progress)
                progress_notes = st.text_area("Notes")
                submit_progress = st.form_submit_button("Submit Progress")

                if submit_progress:
                    new_entry = pd.DataFrame({
                        'Project': [project_name],
                        'Date': [progress_date],
                        'Progress': [progress_value],
                        'Notes': [progress_notes]
                    })
                    st.session_state.progress_data = pd.concat(
                        [st.session_state.progress_data, new_entry], ignore_index=True
                    )
                    st.success("Progress updated successfully!")

    with tab2:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="status-new">🆕</span> Auto Progress Estimation</h3>
            <p>Estimate progress from image analysis using computer vision</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Upload Images for Analysis", expanded=True):
            uploaded_images = st.file_uploader(
                "Upload project images (JPEG, PNG)",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )

            if uploaded_images:
                st.info("Image analysis in progress...")
                progress_bar = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_images):
                    image = Image.open(uploaded_file)
                    progress = analyze_image_progress(image, project_name)

                    # Add to progress data if this is a new progress value
                    project_progress_data = st.session_state.progress_data[
                        st.session_state.progress_data['Project'] == project_name
                        ]
                    if project_progress_data.empty or progress > project_progress_data['Progress'].max():
                        new_entry = pd.DataFrame({
                            'Project': [project_name],
                            'Date': [datetime.now().date()],
                            'Progress': [progress],
                            'Notes': [f"Auto-estimated from image: {uploaded_file.name}"]
                        })
                        st.session_state.progress_data = pd.concat(
                            [st.session_state.progress_data, new_entry], ignore_index=True
                        )

                    progress_bar.progress((i + 1) / len(uploaded_images))
                    time.sleep(0.5)  # Simulate processing time

                st.success("Image analysis completed!")

                # Display uploaded images
                st.subheader("Recently Uploaded Images")
                cols = st.columns(3)
                for idx, img in enumerate(st.session_state.uploaded_images.get(project_name, [])[-6:]):
                    with cols[idx % 3]:
                        st.image(img, caption=f"Image {idx + 1}", use_column_width=True)

    with tab3:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="status-new">🆕</span> Delay Prediction</h3>
            <p>Predict potential delays based on progress trends and historical data</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Delay Prediction Analysis", expanded=True):
            if not project_progress_data.empty:
                delay_prediction = predict_delay(project_progress_data)
                status_color = "green" if "On schedule" in delay_prediction else "orange" if "Slight" in delay_prediction else "red"

                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {status_color};">
                    <h3 style="color: {status_color}; margin-top: 0;">Project Status: {delay_prediction}</h3>
                </div>
                """, unsafe_allow_html=True)

                # Show progress trend
                fig = px.line(
                    project_progress_data,
                    x='Date',
                    y='Progress',
                    title='Progress Trend',
                    markers=True,
                    text='Progress'
                )
                fig.update_traces(textposition="top center")
                fig.add_hline(
                    y=project_data['target_progress'],
                    line_dash="dot",
                    annotation_text="Target",
                    annotation_position="bottom right"
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Progress (%)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show progress statistics
                st.subheader("Progress Statistics")
                progress_stats = project_progress_data['Progress'].describe()
                st.write(progress_stats)

            else:
                st.warning("No progress data available for analysis. Please add progress data first.")

    with tab4:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="status-completed">✅</span> Progress Charts</h3>
            <p>Visualize project progress over time with interactive charts</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Progress Visualization", expanded=True):
            if not project_progress_data.empty:
                # Main progress chart
                fig = px.line(
                    project_progress_data,
                    x='Date',
                    y='Progress',
                    title=f'Progress Over Time - {project_name}',
                    markers=True,
                    text='Progress'
                )
                fig.update_traces(textposition="top center")
                fig.add_hline(
                    y=project_data['target_progress'],
                    line_dash="dot",
                    annotation_text="Target",
                    annotation_position="bottom right"
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Progress (%)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Burnup chart
                st.subheader("Burnup Chart")
                burnup_data = project_progress_data.copy()
                burnup_data['Date'] = pd.to_datetime(burnup_data['Date'], errors='coerce')
                burnup_data['Cumulative Days'] = (burnup_data['Date'] - burnup_data['Date'].min()).dt.days + 1

                fig = px.line(
                    burnup_data,
                    x='Cumulative Days',
                    y='Progress',
                    title='Burnup Chart',
                    markers=True
                )
                fig.add_hline(
                    y=project_data['target_progress'],
                    line_dash="dot",
                    annotation_text="Target",
                    annotation_position="bottom right"
                )
                fig.update_layout(
                    xaxis_title="Days Since Start",
                    yaxis_title="Progress (%)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Progress notes
                st.subheader("Progress Notes")
                st.dataframe(
                    project_progress_data[['Date', 'Progress', 'Notes']].sort_values('Date', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning("No progress data available. Please add progress data first.")

else:
    st.warning("Please select or create a project to get started.")

# Add some sample data if empty (for demo purposes)
if st.session_state.progress_data.empty and st.session_state.projects:
    sample_dates = pd.date_range(
        start=datetime.now().date() - timedelta(days=30),
        end=datetime.now().date(),
        freq='3D'
    )
    sample_progress = np.linspace(10, 70, len(sample_dates))

    for project in st.session_state.projects.keys():
        for date, progress in zip(sample_dates, sample_progress):
            new_entry = pd.DataFrame({
                'Project': [project],
                'Date': [date.date()],
                'Progress': [int(progress)],
                'Notes': [f"Sample data - day {(date.date() - sample_dates[0].date()).days}"]
            })
            st.session_state.progress_data = pd.concat(
                [st.session_state.progress_data, new_entry], ignore_index=True
            )
