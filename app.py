from dash import Dash, dcc, html, dash_table, Input, Output, State
import pandas as pd
import numpy as np
import copy
import os
import io
import base64
from datetime import date
import dash_bootstrap_components as dbc
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# Base dataset to use if no dataset is provided
iris_data = pd.read_csv("Iris.csv", index_col="Id")
base_columns = iris_data.columns

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(
    children=[
        dbc.Row(
            dbc.Col(
                html.H1(children="Automatically analyze your data", className="my-4 py-2"),
                width={"size": 8}
            )
        ),
        html.Br(),
        dbc.Row(
            dbc.Col([
                html.Div("""Welcome to ____ a browser based exploratory data analysis (EDA) and machine learning (ML) demo where you can learn 
                         about the process of ML model development. Here you will walk through the process of initial data evaluation and 
                         processing, finishing with the construction of a model. 
                         """),
                html.Br(),
                html.Div("""If you don't have a dataset but still want to try out the demo, you can use the data included.""")
            ],
            width={"size": 8}),
            justify="left",
            className="p-4"
        ),
        html.Br(),
        dbc.Row(
            dbc.Col(
                html.Div("""Start by uploading your data or use the included data."""),
                width={"size": 8}
            )
        ),
        html.Br(),
        dcc.Store(id="data"),
        dbc.Row(
            dbc.Col(
                dcc.Upload(id="upload-file",
                    children=dbc.Button('Upload File', color="primary")
                ),
                width={"size": 2}
            ),
            class_name="mb-4"
        ),
        html.Br(),
        dbc.Row([
            html.H3("Data Preview", className="pb-4 mt-4"),
            dbc.Col([
                html.P("""The is a temporary placeholder for an explanation about reviewing data. To the right, you will find
                       a sample of your data."""),
                html.Br(),
                html.P("""This is another sentence placehodler to fill in the gap. This is only to make sure I get the sizing correct."""),
                html.Br(),
                html.P("""Here is another placeholder sentence.""")
            ],
            width={"size":8})
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    id="data-table",
                    style={"maxHeight": "800px", "overflow-x": "scroll"}
                )
            )
        ]),
        html.Br(),
        dbc.Card([
            dbc.CardBody(
                html.H3("Data Profile")
            ),
            dbc.ListGroup([
                dbc.ListGroupItem(
                    dbc.Row([
                        dbc.Col("Number of columns:"),
                        dbc.Col(html.Div(id="num-cols"))
                    ])
                ),
                dbc.ListGroupItem(
                    dbc.Row([
                        dbc.Col("Number of rows:"),
                        dbc.Col(html.Div(id="num-rows"))
                    ])
                ),
                dbc.ListGroupItem(
                    dbc.Row([
                        dbc.Col("Data types:"),
                        dbc.Col(html.Div(id="data-types"))
                    ])
                ),
                dbc.ListGroupItem(
                    dbc.Row([
                        dbc.Col("Null count:"),
                        dbc.Col(html.Div(id="null-vals"))
                    ])
                ),
                dbc.ListGroupItem(
                    dbc.Row([
                        dbc.Col("Null % of column:"),
                        dbc.Col(html.Div(id="percent-null"))
                    ])
                )
            ])
        ]),
        html.Div([
            dbc.Row([
                dcc.Graph(id="profile-plots")
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("""When identifying outliers in a dataset, understanding what makes them outliers is crucial. Outliers are data 
                           points significantly different from other observations and may arise from measurement errors, natural variance, 
                           or rare events. Once identified, deciding how to handle outliers depends on their nature and impact on analysis. 
                           Removing them outright can affect data integrity and distort insights, so instead, mitigating their impact by 
                           considering their source is essential. For instance, if an outlier stems from a recording error, correcting 
                           the source might be more appropriate than eliminating the data point."""),
                    html.Br(),
                    html.P("""In data preprocessing, applying transformations can help normalize data and reduce the influence of 
                           outliers on statistical analyses. Techniques like log transformations or winsorization can make data less 
                           sensitive to extreme values without losing critical information. Additionally, modifying data to support 
                           further analysis is crucial. Converting categorical variables into numerical codes, known as encoding, 
                           enables the utilization of algorithms that require numerical inputs. However, it's essential to choose 
                           appropriate encoding methods to avoid introducing unintended biases or misleading the analysis due to the 
                           nature of categorical data. Understanding the source and context of the data assists in making informed 
                           decisions about these transformations, ensuring the integrity and reliability of subsequent analyses."""),
                ], width={"size": 10}
                )
            ],
            justify="center",
            className="p-4"),
            dbc.Row([
                dbc.Col(
                    html.P("""If you're using the base dataset, there aren't any null values present, but if you've uplaoded your own you can choose what to do with those null values here."""),
                width={"size": 10}
                )],
                justify="center",
                className="p-4",
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.DropdownMenu([
                            dbc.DropdownMenuItem("Fill"),
                            dbc.DropdownMenuItem("Drop")
                            ],
                            label="Selection",
                            id="null-dropdown"
                        ),
                        #dbc.Checklist(id="null-checklist", options=["Fill", "Drop"], inline=True, class_name="px-4 py-2"),
                        style={"borderRadius":10}
                    )],
                    width="auto",
                )],
                justify="center",
                className="p-4"
            )
        ]),
        html.Br(),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3("Correlations:"),
                    html.P("""To the right is a heatmap showing the correlations between each of the variables. The correlation values
                            exist on [-1, 1] with -1 being a perfect inverse correlation, and 1 being a perfect correlation"""),
                    html.Br(),
                    html.P("""When reviewing a correlation heatmap, you want to be looking for strong negative or positive correlations.
                           When building models we want predictors (model input variables) to be as independent as possible, ideally completely independent.
                           Many models assume independence and any amount of correlation can bias the results."""),
                    html.Br(),
                    html.P("""Generally, we want the predictors to each carry unique information about the target variable. The more unique they are,
                           the better our model will be. Having multiple variables present all carrying similar information can reduce the overall 
                           effectiveness of the model, and will negatively impact it's interpretation. It does this because the estimates for each
                           predictor coefficient will be impacted by the other variable with shared information leading to imprecise or completely 
                           incorrect estimates."""),
                    html.Br(),
                    html.P("""Looking at the data, are you seeing any variables that are strongly correlated?""")
                ],
                width={"size": 5}),
                dbc.Col(
                    dcc.Graph(id="corr-plot"),
                    width={"size": 5}
                )
            ]),
            dbc.Row(
                dbc.Col([
                    html.P("""There are a few things that can be done to handle strongly correlated variables. Two of the common approaches are
                           dropping one or more of the redundant variables or performing a principal component analysis (PCA). Dropping variables is
                           dont easily and depending on the strength of the correlation, may be all that you need to do. Conducting a PCA
                           is a more advance technique that reduces the dimension of your dataset and retains some specified percentage of
                           of the information."""),
                    html.Br(),
                    html.P("""In this demo, we are going to keep it simple and only remove redundant variables.
                           Are there any variables you'd like to drop?""")
                ])
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.Checklist(id="redundant-checklist", inline=True, class_name="px-4 py-2"),
                        style={"borderRadius":10}
                    ),
                    width="auto"
                )
            ], 
            justify="center",
            class_name="m-4 p-4"),
            dbc.Row(
                html.Div(
                    id="redundant-table",
                    style={"maxHeight": "800px", "overflow-x": "scroll"},
                    className="p-4"
                )
            )
        ]),
        html.Div([
            dbc.Row(
                dbc.Col(
                    html.H3("Modeling", className="m-2"),
                    width={"offset": 1}
                )
            ),
            dbc.Row([
                dbc.Col([
                    html.P("""Machine learning problems broadly fall into four categories based on two primary axes: supervision and unsupervised. 
                           Supervised learning involves training models on labeled data, where inputs are paired with corresponding outputs, 
                           allowing the algorithm to learn patterns and make predictions. It subdivides into two main types: classification 
                           and regression. Classification aims to categorize data into predefined classes or categories, such as spam 
                           detection or image recognition, assigning labels based on features. Conversely, regression predicts continuous 
                           numerical values, like predicting prices or estimating quantities based on input variables."""),
                    html.Br(),
                    html.P("""Unsupervised learning operates on unlabeled data, aiming to derive insights from the data's inherent structure 
                           without predefined outputs. Clustering and dimensionality reduction are common unsupervised learning problems. 
                           Clustering groups similar data points together based on patterns or similarities, aiding in segmentation or 
                           customer profiling. Dimensionality reduction techniques seek to compress data while preserving its essential 
                           features, simplifying complex datasets for easier analysis or visualization. These different problem types in 
                           machine learning cater to diverse data scenarios, enabling the development of models that extract valuable 
                           insights or make accurate predictions across various domains."""),
                ], width={"size": 10}
                )
            ],
            justify="center",
            className="p-4"),
            dbc.Row([
                dbc.Card([
                    dbc.Row([
                        html.Div("This is some placeholder content to see how text will fit on this card along with the tabs", className="p-4 m-4")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Model Type"),
                            dbc.Checklist(
                                options=[
                                    {"label": "Option 1", "value": 1},
                                    {"label": "Option 2", "value": 2},
                                    {"label": "Disabled Option", "value": 3, "disabled": True},
                                ],
                                id="model-switches",
                                switch=True,
                            )
                        ]),
                        dbc.Col([
                            dbc.Label("Target Variable"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Option 1", "value": 1},
                                    {"label": "Option 2", "value": 2},
                                    {"label": "Disabled Option", "value": 3, "disabled": True},
                                ],
                                id="target-switches",
                                switch=True,
                            ),
                        ]),
                        dbc.Col([
                            dbc.Label("Predictors"),
                            dbc.Checklist(
                                id="predictor-switches",
                                switch=True,
                            ),
                        ])
                    ],
                    justify="center",
                    className="p-4"),
                    dbc.Tabs([
                        dbc.Tab([
                            dbc.Row(
                                dbc.Col(
                                    html.Div("""Ordinary Least Squares Regression (OLS) is a statistical method used for predictive modeling and identifying 
                                    relationships between variables. Its key assumptions include linearity between the dependent and independent variables, 
                                    homoscedasticity (constant variance of errors), independence of observations, and normality of error distribution. 
                                    OLS is straightforward, provides coefficient estimates and insights into variable relationships, yet it can be 
                                    sensitive to outliers and violations of its assumptions, potentially impacting the accuracy and reliability of 
                                    its predictions.""", className="p-4"),
                                    width={"size": 10}
                                ),
                                justify="center"
                            ),
                            dbc.Row([
                                dcc.Graph()
                            ])
                        ], label="Ordinary Least Squares Regression", className="p-2"),
                        dbc.Tab([
                            dbc.Row(
                                dbc.Col(
                                    html.Div("""XGBoost is an ensemble learning technique renowned for its predictive power in regression and classification tasks. 
                                            It's based on boosting, sequentially combining weak learners into a robust model. XGBoost assumes its weak learners 
                                            (decision trees) are relatively simple, aiming to minimize errors in a step-by-step fashion. Its strengths lie in 
                                            handling complex interactions among variables, regularization to prevent overfitting, and feature importance estimation. 
                                            However, it might require more parameter tuning, and its interpretability can be challenging due to its complex structure""", className="p-4"),
                                    width={"size": 10}
                                ),
                                justify="center"
                            ),
                            dbc.Row([
                                dcc.Graph()
                            ])
                        ], label="XGBoost", className="p-2"),
                        dbc.Tab([
                            dbc.Row(
                                dbc.Col(
                                    html.Div("""K-means++ is an improvement over the traditional K-means clustering algorithm, focusing on better initializing 
                                            centroids for more accurate cluster formation. Its assumption lies in minimizing the intra-cluster variance by 
                                            choosing initial centroids intelligently. The benefits include improved convergence speed and more robust cluster 
                                            assignments. However, it can still converge to suboptimal solutions depending on initializations, and it assumes 
                                            clusters as spherical and of similar sizes, impacting performance with irregularly shaped or varied-sized clusters.""", className="p-4"),
                                    width={"size": 10}
                                ),
                                justify="center"
                            ),
                            dbc.Row([
                                dcc.Graph()
                            ])
                        ], label="K-means++",  className="p-2")
                    ])
                ])
            ])
        ], className="p-4 m-4")
        
        #html.Div(
        #    children=[
        #        html.Div(
        #            children=[
        #                html.Div("Ticker", className="menu-title"),
        #                dcc.Dropdown(id="ticker-filter")
        #            ]
        #        ),
        #        html.Div(
        #            children=[
        #                html.Div("Column", className="menu-title"),
        #                dcc.Dropdown(id="column-filter")
        #            ]
        #        )
        #    ], className="menu"
        #),
        #html.Div(children=[
        #    html.H1("Chart"),
        #    dcc.Graph(id="chart", config={"displayModeBar": False})
        #])
        # Add drop down menu with all the unique tickers available
        # Tickbox selector for each column to be displayed on the plot
        # Add a data range selector
        # Lineplot
    ]
)

@app.callback(Output("data", "data"),
              Input("upload-file", "contents"),
              Input("upload-file", "filename"))
def read_data(contents, filename):
    # Upload data using button and store for use
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=',')
            return df.to_dict("records")
        
@app.callback(Output("data-table", "children"),
              Input("data", "data"))
def build_table(data):
    if data:
        df = pd.DataFrame.from_dict(data)
        return dbc.Table.from_dataframe(df=df.head(), striped=True, bordered=True, hover=True)
    else:
        return dbc.Table.from_dataframe(df=iris_data.head(), striped=True, bordered=True, hover=True)

@app.callback(Output("num-cols", "children"),
              Output("num-rows", "children"),
              Output("data-types", "children"),
              Output("null-vals", "children"),
              Output("percent-null", "children"),
              Input("data", "data"))
def data_profile(data):
    if data:
        df = pd.DataFrame.from_dict(data)
        num_cols = df.shape[1]
        num_rows = df.shape[0]
        data_types = df.dtypes.to_dict()
        null_vals = df.isnull().sum().to_dict()
        percent_null = np.round(df.isnull().sum() / df.shape[0]).to_dict()
    else:
        num_cols = iris_data.shape[1]
        num_rows = iris_data.shape[0]
        data_types = iris_data.dtypes.to_dict()
        null_vals = iris_data.isnull().sum().to_dict()
        percent_null = np.round(iris_data.isnull().sum() / iris_data.shape[0]).to_dict()
    
    return (num_cols, num_rows, dict_to_div(data_types), dict_to_div(null_vals), dict_to_div(percent_null))
    
@app.callback(Output("profile-plots", "figure"),
              Input("data", "data"))
def profile_plots(data):
    if data:
        df = pd.DataFrame.from_dict(data)
        ncols = 4
        nrows = (len(df.columns) // ncols) + 1

        # loop over columns and create plot
        titles = [col for col in df.columns]
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)

        col_loc = 0
        for row in range(1, nrows + 1):
            for i in range(1, ncols + 1):
                if col_loc == len(df.columns):
                    break

                fig.add_trace(go.Histogram(x=df[df.columns[col_loc]], showlegend=False, name=""), row=row, col=i)
                col_loc += 1

    else:
        ncols = 4
        nrows = (len(iris_data.columns) // ncols) + 1

        # loop over columns and create plot
        titles = [col for col in iris_data.columns]
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)

        col_loc = 0
        for row in range(1, nrows + 1):
            for i in range(1, ncols + 1):
                if col_loc == len(iris_data.columns):
                    break

                fig.add_trace(go.Histogram(x=iris_data[iris_data.columns[col_loc]], showlegend=False, name=""), row=row, col=i)
                col_loc += 1
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False, visible=False, showticklabels=False) # Hide x axis ticks 
    fig.update_yaxes(showgrid=False, visible=False, showticklabels=False) # Hide y axis ticks

    return fig


@app.callback(Output("corr-plot", "figure"),
              Input("data", "data"))
def make_heatmap(data):
    pio.templates.default = "plotly_white"
    if data:
        df = pd.DataFrame.from_dict(data)
        df = cat_to_code(df)
        corr = df.corr()
        mask = np.triu(np.ones(corr.shape)).astype(bool)
        corr = corr.where(mask).T
    else:
        iris = cat_to_code(iris_data)
        corr = iris.corr()
        mask = np.triu(np.ones(corr.shape)).astype(bool)
        corr = corr.where(mask).T

    return {
        "data": [
            go.Heatmap(
                z=corr,
                x=corr.columns,
                y=corr.columns,
                colorscale=px.colors.diverging.RdBu,
                zmin=-1,
                zmax=1
            )
        ],
        "layout":
            go.Layout(
                width=900,
                height=600,
                yaxis_autorange='reversed',
                yaxis={"showgrid": False, "visible": False, "showticklabels": False},
                xaxis={"showgrid": False, "visible": False, "showticklabels": False}
            )
        }

"""@app.callback(Output("null-dropdown", "label"),
              Input("null-dropdown", "value"))
def null_dropdown(value):
    print(value)"""

@app.callback(Output("redundant-checklist", "options"),
              Input("data", "data"))
def list_vars(data):
    if data:
        df = pd.DataFrame.from_dict(data)
        return df.columns
    else:
        return base_columns

@app.callback(Output("redundant-table", "children"),
              Input("redundant-checklist", "value"),
              Input("data", "data"))
def remove_redundant(value, data):
    if data:
        df = pd.DataFrame.from_dict(data)
        if value:
            df.drop(value, axis=1, inplace=True)
        return dbc.Table.from_dataframe(df=df.head(), striped=True, bordered=True, hover=True)
    else:
        df = copy.copy(iris_data)
        if value:
            df.drop(value, axis=1, inplace=True)
        return dbc.Table.from_dataframe(df=df.head(), striped=True, bordered=True, hover=True)


@app.callback(Output("target-switches", "options"),
              Input("data", "data"))
def create_target_options(data):
    if data:
        df = pd.DataFrame.from_dict(data)
        return [{"label":col, "value":i+1, "disabled":False} for i, col in enumerate(df.columns)]
    else:
        return [{"label":col, "value":i+1, "disabled":False} for i, col in enumerate(iris_data.columns)]
    
@app.callback(Output("predictor-switches", "options"),
              Output("predictor-switches", "value"),
              Input("data", "data"),
              Input("target-switches", "value"))
def update_predictor_options(data, value):
    if data:
        df = pd.DataFrame.from_dict(data)
        options = []
        for i, col in enumerate(df.columns):
            option = {"label":col, "value":i+1, "disabled":False}
            if value and i+1 == value:
                option["disabled"] = True
                # Remove target from value list
            options.append(option)
    else:
        options = []
        for i, col in enumerate(iris_data.columns):
            option = {"label":col, "value":i+1, "disabled":False}
            if value and i+1 == value:
                option["disabled"] = True
                # Remove target from value list
            options.append(option)
    return (options, [])
        


###############################################################################
# Utilities
###############################################################################

def dict_to_div(dic):
    return [html.Div(f"{key} - {dic[key]}") for key in dic]

def cat_to_code(data):
    df = pd.DataFrame.from_dict(data)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
    return df

if __name__ == "__main__":
    app.run_server(debug=True)