# Airline Passenger Satisfaction Classification Project ✈️

In this project, you will apply classification techniques to the ****Airline Passenger Satisfaction**** dataset to predict whether a passenger is satisfied with their flight experience. The dataset contains various features related to passengers, including their demographics, flight information, and survey responses. Your goal is to build a classification model that predicts passenger satisfaction (satisfied or dissatisfied) based on these features.

The dataset file will be available in the ****GitHub repository**** for this project, so please clone the repository to access it.

* * *

### ****Dataset Overview:****

The ****Airline Passenger Satisfaction**** dataset contains survey data collected from airline passengers. It includes features such as flight distance, passenger age, seat comfort, and more, with the target variable being the passenger's satisfaction level (satisfied or dissatisfied).

<div style="height:115px;background-color:#f0f9ff;padding:16px;border:1px solid #b3e0ff;border-radius:8px;color:#333333;margin:10px;white-space:normal">
    <span style="font-weight:bold;color:#0056b3;white-space:normal">🗃️ Dataset Reference:</span> 
    <span style="display:block;white-space:normal">
        🌐 <a href="https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data" target="_blank" style="color:#007bff;text-decoration:none">https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data</a>
    </span>
    <span style="display:block;color:#666666;white-space:normal">
        📅 Accessed: Feb 2025
    </span>
</div>

****Dataset file:****
`airline_passenger_satisfaction.csv`

* * *

### ****Project Instructions:****

This project will be completed in two parts: ****Exploratory Data Analysis (EDA) and Data Cleaning**** followed by ****Classification Model Building and Evaluation****.

* * *

#### ****Part 1: Exploratory Data Analysis (EDA) and Data Cleaning 🧑‍🔬****

****Objective:****

-   Understand the dataset and perform necessary cleaning and preprocessing to prepare it for modeling.
****Deliverables for Part 1:****

1.  ****EDA.ipynb notebook****:
2.  -   This should contain all your code, visualizations, and explanations for the exploratory analysis and cleaning process.
    -   Tasks to complete:
    -   -   Load the dataset (`airline_passenger_satisfaction.csv`).
        -   Inspect the data using `head()`, `info()`, and `describe()` functions.
        -   Analyze distributions of key features
        -   Identify and visualize missing values and handle them (impute or remove).
        -   Detect and handle outliers in numerical features (box plots, histograms).
        -   Perform feature engineering.
        -   Save the cleaned dataset as `cleaned_airline_passenger_satisfaction.csv` for use in the next part.
3.  ****cleaned\_airline\_passenger\_satisfaction.csv file****:
4.  -   The cleaned dataset ready for modeling.
    -   This file should be saved after handling missing values, outliers, and applying any necessary feature engineering.
* * *

#### ****Part 2: Classification Model Building and Evaluation 📈****

****Objective:****

-   Build and evaluate a classification model to predict passenger satisfaction.
****Deliverables for Part 2:****

1.  ****modeling.ipynb notebook****:
2.  -   This notebook should contain the code for building, training, and evaluating the classification model.
    -   Tasks to complete:
    -   -   Load the cleaned dataset (`cleaned_airline_passenger_satisfaction.csv`).
        -   Split the data into training and testing sets (typically an 80/20 or 70/30 split).
        -   Select relevant features and perform any additional necessary feature engineering.
        -   Train a classification model (Logistic Regression, Random Forest, or Support Vector Machine).
        -   Evaluate the model using appropriate metrics.
        -   Track any improvements made to the model through hyperparameter tuning or feature adjustments.
* * *

#### ****Part 3: Interactive Presentation with Streamlit App 🎨****

****Objective:****

-   Present your findings and model performance interactively using a ****Streamlit app****.
****Deliverables for Part 3:****

1.  ****Streamlit app****:
2.  -   Build and deploy a ****Streamlit app**** that:
    -   -   Allows users to interact with your model and make predictions based on new data (e.g., inputting passenger features like seat comfort, age, flight distance).
        -   Displays key insights from the exploratory analysis, including visualizations of important trends.
        -   Shows the model's performance metrics in an engaging format.
        -   Provides recommendations or insights on how airlines could improve passenger satisfaction.
    -   Make the app user-friendly and visually appealing.
    -   Ensure the app provides a smooth experience for users to interact with the model and view results.
3.  ****Deploy the Streamlit app****:
4.  -   Deploy the Streamlit app to ****Streamlit Cloud****.
    -   Provide a link to the deployed app in your README.md file for easy access.
* * *

### ****Deliverables Summary:****

1.  ****EDA.ipynb notebook****:
2.  -   Contains all the code, visualizations, and explanations for exploratory data analysis and data cleaning.
3.  ****cleaned\_airline\_passenger\_satisfaction.csv file****:
4.  -   The cleaned dataset that has been preprocessed and is ready for modeling.
5.  ****modeling.ipynb notebook****:
6.  -   Contains the code for building, training, and evaluating the classification model.
7.  ****cleaned\_airline\_passenger\_satisfaction.csv file****:
8.  -   The cleaned dataset that was used in the modeling process (ensure this is included in the modeling part as well).
9.  ****Presentation 🎤 (5 minutes)****:
10.  -   Prepare and save a ****PDF presentation**** that highlights your analysis and model performance.
11.  ****README.md file****:
12.  -   A ****README.md**** file summarizing your project, the steps taken, model performance, and recommendations.
13.  ****Streamlit app****:
14.  -   An interactive ****Streamlit app**** showcasing your findings, model performance, and recommendations.
    -   Deploy the app and provide a link in your README.md file.
* * *

### ****Important Notes:****

-   The ****Streamlit app**** will provide a dynamic way to interact with your analysis and model, making the results accessible and easy to understand for any user.
-   Ensure that the ****Streamlit app**** includes both an explanation of your key findings and an interactive section where users can input their own data (e.g., passenger details) to get predictions from your model.
-   Save the app link in your README.md to allow easy access for others to interact with it.
* * *

This project will allow you to demonstrate your skills in data cleaning, classification modeling, and presenting actionable insights that can be valuable for improving passenger satisfaction. Make sure to document your process well and provide clear, engaging visualizations in your notebook and presentation.

* * *

### ****GitHub Repository Instructions 📂:****

-   Clone the ****GitHub repository**** to begin working on the project. The repository link is provided here:
    [****GitHub Repository Link****](https://github.com/Data-Analytics-Flex/M8-Project-Airline-Satisfaction-Predictions)
    To clone the repository to your local machine, use the following command:

    git clone <repository\_link\>

-   After cloning, you can start by working through the dataset, following the instructions, and implementing the analysis in your Jupyter Notebook.
-   Once completed, push your changes to the repository, including:
-   -   The Jupyter notebook with analysis and visualizations.
    -   The presentation saved as a PDF.
    -   The README.md file.

Good luck with your analysis and modeling process! 🚀

* * *

## 📝 Rubric

The grading for this project will be based on three primary categories: ****Functional****, ****Interpersonal****, and ****Technical****. Each category includes specific dimensions that will be evaluated on a scale from 1 to 5. To pass the project, you must score ****3 or higher**** in each component.

## ****Functional****

#### ****Value/Impact****

****Rubric Component****: Evaluate how well the project identifies insights into customer satisfaction and presents actionable recommendations.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Deep insights that are highly relevant and actionable for business or operational improvements.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Clear, actionable insights that could inform business decisions.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Some insights with basic actionable recommendations.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Limited insights, lacks actionable recommendations.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">No meaningful insights or actionable recommendations.</p></td></tr></tbody></table>

* * *

#### ****Requirements****

****Rubric Component****: Ensure the project includes data cleaning, EDA, predictive modeling, evaluation, and Streamlit app for presentation.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 774px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 774px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Fully integrated components with excellent execution and coverage of the full project scope.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 774px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">All components included, completed thoroughly with attention to detail.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 774px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">All components present but with some gaps in detail or execution.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 774px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Some components missing or partially incomplete.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 774px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Missing significant components, incomplete work.</p></td></tr></tbody></table>

* * *

#### ****Timelines****

****Rubric Component****: Evaluate timeliness of delivery and project completion within scope.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Ahead of schedule, with detailed planning and flawless execution.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Completed on time with organized planning and clear progress tracking.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">On time but rushed or incomplete.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Late with minimal explanation or planning.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 771px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Significantly late, no clear timeline management.</p></td></tr></tbody></table>

* * *

## ****Interpersonal****

#### ****Professionalism****

****Rubric Component****: Evaluate how well the notebook, Streamlit app, and report reflect professionalism through clarity, tone, and structure.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 779px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 779px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Highly polished, engaging, and professional tone, with clarity in every section.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 779px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Well-organized, professional, and clearly written throughout.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 779px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Clear and professional tone, but needs refinement in clarity.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 779px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Some professional tone, but lacks clarity or structure.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 779px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Unprofessional tone, unclear organization.</p></td></tr></tbody></table>

* * *

#### ****Presentation****

****Rubric Component****: Evaluate the organization, clarity, and visual appeal of the Streamlit app and presentation.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 775px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 775px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Highly engaging, visually appealing, and intuitive Streamlit app with strong flow and clarity.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 775px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Well-organized, clear, and visually appealing Streamlit app and presentation.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 775px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Clear and structured, but lacks visual polish or engagement.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 775px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Somewhat organized with minimal visual appeal or clarity.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 775px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Disorganized or lacks visual appeal, difficult to follow.</p></td></tr></tbody></table>

* * *

#### ****Feedback****

****Rubric Component****: Evaluate how the student integrates feedback (if provided) into the project to improve it.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Fully integrates feedback, showing significant improvements and learning from it.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Actively engages with feedback and integrates changes.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Incorporates some feedback with moderate improvements.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Minimal engagement with feedback.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">No engagement with feedback.</p></td></tr></tbody></table>

* * *

## ****Technical****

#### ****Complexity****

****Rubric Component****: Evaluate the depth and complexity of the exploratory data analysis, cleaning, and modeling processes.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Deep, advanced analysis, and modeling with multiple advanced techniques.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Comprehensive EDA and modeling with detailed and complex analyses.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Solid analysis and modeling with some complexity.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Some analysis and modeling, but lacks depth or complexity.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 778px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Very basic analysis or modeling, minimal complexity.</p></td></tr></tbody></table>

* * *

#### ****Design****

****Rubric Component****: Evaluate the clarity and structure of the notebooks, including well-commented code, markdown explanations, and clear visualizations.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 776px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 776px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Exceptionally organized, well-commented notebooks, clear explanations, and polished visualizations.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 776px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Well-organized notebooks, with clear code explanations and visuals.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 776px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Clear structure, but could benefit from more explanations or visual polish.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 776px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Somewhat organized but lacks clarity in explanations or visuals.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 776px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Poorly organized notebooks with unclear code or visuals.</p></td></tr></tbody></table>

* * *

#### ****Reliability****

****Rubric Component****: Evaluate the accuracy of the data cleaning, modeling processes, and how the models are evaluated and compared.

<table class="editor-table"><colgroup><col><col></colgroup><tbody><tr><th class="editor-table-cell editor-table-cell-header" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Rating</p></th><th class="editor-table-cell editor-table-cell-header" style="width: 772px; border: 1px solid black; vertical-align: top; text-align: start; background-color: rgb(242, 243, 245);"><p class="editor-paragraph">Description</p></th></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">5</p></td><td class="editor-table-cell" style="width: 772px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Highly reliable and accurate work, with detailed, well-documented model performance evaluation.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">4</p></td><td class="editor-table-cell" style="width: 772px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Accurate, reliable data cleaning, modeling, and thorough model evaluation.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">3</p></td><td class="editor-table-cell" style="width: 772px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Generally accurate work, but lacks full consistency or clarity in evaluation.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">2</p></td><td class="editor-table-cell" style="width: 772px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Some errors in cleaning, modeling, or model comparison.</p></td></tr><tr><td class="editor-table-cell" style="border: 1px solid black; width: 350px; vertical-align: top; text-align: start;"><p class="editor-paragraph">1</p></td><td class="editor-table-cell" style="width: 772px; border: 1px solid black; vertical-align: top; text-align: start;"><p class="editor-paragraph">Numerous errors in data cleaning, model performance, or evaluation.</p></td></tr></tbody></table>

****Passing Criteria:****

-   ****Minimum Requirement****: A score of ****3 or higher**** in each component to pass.
-   ****Total Points/Percentages****: These are provided for informational purposes but do not affect the passing criteria.
