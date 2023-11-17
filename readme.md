Automatically analyize your data in browser.

Welcome to ____ a browser based exploratory data analysis (EDA) and machine learning (ML) demo where you can learn about the process of ML model development. Here you will walk through the process of initial data evaluation and processing, finishing with the construction of a model.

If you don't have a dataset but still want to try out the demo, you can use the data included.

Start by uploading your data (use the included data).

[Upload Data Button]

Data Preview:
- What to look for upon first review of your data.

Data Profile:
- data profile table containing some core info
    - number of columns
    - number of rows
    - column data type
    - number of missing values
    - % of column with missing values
- Explanation about evaluataing each variable to find anything particularl interesting or unusual
- Plots for each variable showing:
    - distribution
    - min, max, mean, median
- Explanation about what sort of things you might consider doing to the variables based on what has been learned so far.
- Talk about:
    - finding outliers and what makes them outliers
    - what to do with outliers if you find any and why it's important to consider the source of the data to help inform decision making
    - applying transformations
    - modifying the data in appropriate ways to support other analysis such as converting categorical variables to a code because you can't use string representation
- Interactivity:
    - allow user to make a decision about how to fill missing values, or to remove the rows entirely
    - whether or not the user wants to remove any detected outliers and explain how these outliers are detected

Data Relationships:
- Explanation about why it's important to check for correlations between variables
    - talk about how many models assume independence between predictors, etc
- Create a heatmap displaying the correlations between all the variables, prefer a triangle over complete square
- Mention that any categorical variables had to be changed to integers in order


Modelling:
- Describe the types of modelling problems i.e. regression, classification and how the available models are used to solve them
- Build accordian with a description of each model, and some assumptions about them i.e. linear relationships, variable independence, scaling of input to [0,1], etc
    - NOTE: when talking about linear relationship, this is a good time to include some brief details referencing transformations that can be applied to the variables.
- Create a checklist for each of: model type, target variable, predictors
- Create a plot for each model type that best shows off the performance of the model
- Create a brief table with some of the measurements used to determine model performance
- Model interpretation:
    - explanation on how to interpret the results of each model, in a general sense since I cannot say exactly what's going on for each dataset

1. Upload data
    1. Provide data profile including: 
        - number of columns
        - number of rows
        - column data type
        - number of missing values
        - % of column with missing values
2. Clean data
    1. How to handle missing values
3. Perform covariate analysis for columns in data
    1. Correlation coefficient plot
    2. Mutual information coefficient plot
4. Create data visualizations
    1. Selector for variable to be plotted
    2. Selector for plot top, line, bar, historgram, pie, etc
5. Create model
    1. Select model type
    2. Select target variable
    3. Select variables to use as predictors


