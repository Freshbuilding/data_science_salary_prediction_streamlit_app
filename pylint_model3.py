'''
This program purpose is to create an application
to present insights on Data science jobs.
It can also predict data science salaries.
'''

# import packages
import pickle as pkl
import streamlit as st
import pandas as pd
import plotly.express as px

# import data and model

# Load the dataset
salaries = pd.read_csv('ds_salaries.csv')

# Load the saved model
with open(
    'model_2023.sav',
        'rb') as f:
    model = pkl.load(f)

# Preprocessing user input

def preprocess_inputs(title, experience, remoter, size, location):
    """
    This function takes in user inputs related to a job posting
    and preprocesses them to be used for model predictions.
    """
    user_input_dict = {
        'job_title': [title], 
        'experience_level': [experience],
        'remote_ratio': [remoter], 
        'company_size': [size], 
        'company_location_is_US': [location]
    }

    user_input = pd.DataFrame(data=user_input_dict)

    cleaner_type = {
        'job_title': {
            'Data Analyst': 0,
            'Data Scientist': 1,
            'Data Engineer': 2,
            'Machine Learning Engineer': 3
        },
        'experience_level': {
            'Entry-level': 0,
            'Mid-Level': 1,
            'Senior': 2
        },
        'remote_ratio': {
            'No remote': 0,
            'Semi remote': 1,
            'Full remote': 2
        },
        'company_size': {
            'Small': 0,
            'Medium': 1,
            'Large': 2
        },
        'company_location_is_US': {
            'US': 1,
            'Other': 0
        }
    }

    user_input = user_input.replace(cleaner_type)

    return user_input


# Function to create a bar chart


def create_bar_chart(data_frame, column, title):
    '''function to create a bar chart'''
    counts = data_frame[column].value_counts().head(5)
    fig = px.bar(
        counts,
        x=counts.index,
        y=counts.values,
        color=counts.index,
        text=counts.values,
        labels={
            'index': column,
            'y': 'count',
            'text': 'count'},
        template='seaborn',
        title=f'<b>{title}')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def predict_salary(ml_model, preprocessed_input):
    """ This function takes in a model and preprocessed input """
    pred = ml_model.predict(preprocessed_input)
    # extract the scalar value from the numpy array
    pred_value = pred[0][0]
    return pred_value

# Page layout


def main():
    '''function to create the main page'''

    # main page layout

    # Set Page Configuration
    st.set_page_config(
        page_title='Exploring Computer Science Careers: Salaries, Jobs, and Global Trends',
        layout='wide')

    # Define the page title text
    title_text = "<h1 style='text-align: center;'>Exploring Computer Science Careers:\
          Salaries, Jobs, and Global Trends</h1>"

    # Display the page title with centered styling
    st.markdown(title_text, unsafe_allow_html=True)

    # Set Header
    st.write(
        "The field of data science has witnessed a rapid expansion in recent years,\
             owing to the availability of vast amounts of data and the development of\
             sophisticated tools to analyze it. As a result, there is a growing demand\
             for data scientists across various industries. These professionals are\
             responsible for extracting insights and making data-driven decisions that\
             can impact a company's bottom line. To prepare for a data science job,\
             candidates need to have a strong foundation in statistics, programming,\
             and machine learning techniques. This app aims to provide candidates with\
             the necessary knowledge and understanding of the data science sector,\
             its requirements, and the skills needed to succeed in this field.")

    st.markdown(
        "<h2 style='text-align: center;'>Insights on computer science jobs</h2>",
        unsafe_allow_html=True)

    # Call the function for each tab
    tab1, tab2, tab3, tab4 = st.tabs(
        ['Most popular roles in Data Science',
         'Most represented company location',
         'Highest paid data science jobs',
         'Company Sizes in Data Science Field'])

    with tab1:
        create_bar_chart(
            salaries,
            'job_title',
            'Most popular roles in Data Science')

    with tab2:
        create_bar_chart(
            salaries,
            'company_location',
            'Most represented company location')

    with tab3:
        hp_jobs = salaries.groupby(
            'job_title', as_index=False)['salary_in_usd'].max().sort_values(
            by='salary_in_usd', ascending=False).head(10)
        fig = px.bar(
            hp_jobs,
            x='job_title',
            y='salary_in_usd',
            color='job_title',
            labels={
                'job_title': 'job title',
                'salary_in_usd': 'salary in usd'},
            template='seaborn',
            text='salary_in_usd',
            title='<b>Highest paid data science jobs')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        create_bar_chart(
            salaries,
            'company_size',
            'Company Sizes in Data Science Field')

    st.image(
        'iStock-1221293664-1-1-1.jpg',
        use_column_width=True)

    # sidebar layout

    st.sidebar.image(
        'ab.png',
        width=150)
    st.sidebar.title('Predict your future salary')
    st.sidebar.write('Enter your profile information below:')

    # inputs
    title = st.sidebar.selectbox(
        'Please choose your job title',
        ('Data Analyst',
         'Data Scientist',
         'Data Engineer',
         'Machine Learning Engineer'))
    experience = st.sidebar.selectbox(
        'Please choose your experience level',
        ('Entry-level',
         'Mid-Level',
         'Senior'))
    remoter = st.sidebar.selectbox(
        'Please choose your remote ratio',
        ('No remote',
         'Semi remote',
         'Full remote'))
    size = st.sidebar.selectbox(
        'Please choose the company size',
        ('Small',
         'Medium',
         'Large'))
    location = st.sidebar.radio(
        'Select Location:',
        ('US',
         'Other'))

    # Pre-processing user_input
    user_input = preprocess_inputs(title, experience, remoter, size, location)

    # predict button
    if st.sidebar.button("Predict"):
        pred_value = predict_salary(model, user_input)
        st.sidebar.write(f"Prediction: {pred_value:.2f} USD")


if __name__ == '__main__':
    main()
