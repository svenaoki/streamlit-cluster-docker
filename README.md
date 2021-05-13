# Streamlit app to classify or cluster the Iris flower dataset

This app allows to cluster or classify the Iris flower dataset. It also allows to choose which algorithm to use, their specifications and the size of train and test sets.
For convenience a Dockerimage is provided, too.

To run the app using docker, follow guidelines below:
Install Docker
Fork the folder
Open Powershell and navigate to the directory
a. docker build -t yourImageName:ImageVersion .
b. docker run -d 8501:8501 yourImageName:ImageVersion
That's it and the app should run on localhost:8501.
The port can be altered in the Dockerfile, and then consequently in the cmd.


![alt text](https://github.com/svenaoki/streamlit-cluster-docker/blob/main/docs/application.PNG)
