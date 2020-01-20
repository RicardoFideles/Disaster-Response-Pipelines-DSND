# Disaster Response Pipeline Project

### Motivation

Using learned knowledge and built on data engineering skills to expand opportunities and potential as a data scientist. 
In this project, the goal It's to aplly these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Dataset

The data set containing real messages that were sent during disaster events. 

### The Project

The taks it was to build machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
This project aldo include a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data.

![alt text](./images/disaster-response-project1.png)

![alt text](./images/disaster-response-project2.png)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
