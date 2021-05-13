# Streamlit app to classify or cluster the Iris flower dataset

This app allows to cluster or classify the Iris flower dataset. It also allows to choose which algorithm to use, their specifications and the size of train and test sets. Given the nature of Streamlit, or React, the dashboards builds up and re-renders with almost any change during interaction.
For convenience a Dockerimage is provided, too.

To run the app using docker, follow the guidelines below:
1) Install Docker  
2) Fork the folder
3) Open Powershell and navigate to the directory<br/>
  a. docker build -t yourImageName:ImageVersion .<br/>
  b. docker run -d 8501:8501 yourImageName:ImageVersion


![alt text](https://github.com/svenaoki/streamlit-cluster-docker/blob/main/docs/application.PNG)
