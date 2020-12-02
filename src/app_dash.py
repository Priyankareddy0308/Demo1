from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
import pandas as pd


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names].values

def make_full_pipeline1(DP_train):
    #df = pd.read_csv('../data/Customer-Value-Analysis.csv').set_index('Customer')
    X = DP_train.drop(['Disease'], axis=1)
    y = DP_train['Disease']

    cats = [var for var, var_type in X.dtypes.items() if var_type == 'object']
    nums = [var for var in X.columns if var not in cats]
    # Defining the steps in the categorical pipeline
    cat_pipeline = Pipeline([('cat_selector', FeatureSelector(cats)),
                             ('one_hot_encoder', OneHotEncoder(sparse=False))])

    # Defining the steps in the numerical pipeline
    num_pipeline = Pipeline([
        ('num_selector', FeatureSelector(nums)),
        ('std_scaler', StandardScaler()),
    ])

    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                                   ('cat_pipeline', cat_pipeline)]
                                 )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # fit and transform the custom transformer in train
    _ = full_pipeline.fit_transform(X_train)
    return full_pipeline

#########################
import os
# We start with the import of standard ML librairies
import pandas as pd
import math
from joblib import load
#from data_processing import make_full_pipeline1
import pickle
# We add all Plotly and Dash necessary librairies
import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

HOME_PATH = os.getcwd()
DATA_PATH = os.path.join(HOME_PATH, 'data')
MODELS_PATH = os.path.join(HOME_PATH, 'model')
ASSETS_PATH = os.path.join(HOME_PATH, 'assets')

DP_train = pd.read_csv(os.path.join(DATA_PATH, "Disease Prediction Training.csv"))
sk_best = load(os.path.join(MODELS_PATH, 'best.joblib'))

#full_pipeline = load(os.path.join(MODELS_PATH, 'transformer.joblib'))
full_pipeline = make_full_pipeline1(DP_train)
#ohe_path = os.path.join(MODELS_PATH, 'ohe_categories.pkl')
perfs_path = os.path.join(MODELS_PATH, 'sk_best_performances.pkl')

#with open(ohe_path, 'rb') as input:
    #ohe_categories = pickle.load(input)

#categories = []
#for k, l in ohe_categories.items():
    #categories.append([f'{k}_{cat}' for cat in list(l)])
#flatten = lambda l: [item for sublist in l for item in sublist]
#categories = flatten(categories)

with open(perfs_path, 'rb') as input:
    perfs = pickle.load(input)
# scaling

cats = [var for var, var_type in DP_train.dtypes.items() if var_type == 'object']
nums = [var for var in DP_train.columns if var not in cats]
nums.remove('Disease')

TOP = 10
# We create a DataFrame to store the features' importance and their corresponding label
df_feature_importances = pd.DataFrame(sk_best.feature_importances_ * 100, columns=["Importance"],
                                      index=nums + cats)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)
df_feature_importances = df_feature_importances.loc[df_feature_importances.index[:TOP]]

# We create a Features Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                  )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)

# We create a Features perfomances Bar Chart
fig_perfs = go.Figure()
fig_perfs.add_trace(go.Bar(y=list(perfs.keys()),
                           x=list(perfs.values()),
                           marker_color='rgb(171, 226, 251)',
                           orientation='h')
                    )
fig_perfs.update_layout(title_text='<b>Best Model Performances<b>', title_x=0.5)

cat_children = []
for var in cats:
    # Categorical children
    sorted_modalities = list(DP_train[var].value_counts().index)
    cat_children.append(html.H4(children=var))
    cat_children.append(dcc.Dropdown(
        id='{}-dropdown'.format(var),
        options=[{'label': value, 'value': value} for value in sorted_modalities],
        value=sorted_modalities[0]
    ))

linear_children = []
for var in nums:
    # linear children
    linear_children.append(html.H4(children=var))
    desc = DP_train[var].describe()
    linear_children.append(dcc.Slider(
        id='{}-dropdown'.format(var),
        min=math.floor(desc['min']),
        max=round(desc['max']),
        step=None,
        value=round(desc['mean']),
        marks={i: '{}°'.format(i) for i in
               range(int(desc['min']), int(desc['max']) + 1, max(int((desc['std'] / 1.5)), 1))}
    ))
    
DT = []
var = grid.best_estimator_
for hps, values in grid.best_params_.items():
    # linear children
    DT.append(html.H4(children=(hps, values)))
    DT.append(dcc.Slider(
        id='{}-dropdown'.format(values),
        min=1,
        max=10,
        step=1,
        value=4
    ))
# The command below can be activated in a standard notebook to display the chart
# fig_features_importance.show()


app = dash.Dash(__name__,
# external CSS stylesheets
 external_stylesheets = [
     #"https://raw.githack.com/Athena75/IBM-Customer-Value-Dashboarding/main/assets/style.css",
     "https://rawcdn.githack.com/Athena75/IBM-Customer-Value-Dashboarding/df971ae38117d85c8512a72643ce6158cde7a4eb/assets/style.css"
 ]
)

# We apply basic HTML formatting to the layout
app.layout = html.Div(children=[
    # first row : Title
    html.Div(children=[
        html.Div(children=[html.H1(children="Simulation Tool : IBM Customer Churn")],
                 className='title'),

    ],
        style={"display": "block"}),
    # second row :
    html.Div(children=[
        # first column : fig feature importance + linear + prediction
        html.Div(children=[
            html.Div(children=[dcc.Graph(figure=fig_features_importance, className='graph')] + linear_children),
            # prediction result
            html.Div(children=[html.H2(children="Prediction:"),
                               html.H2(id="prediction_result")],
                     className='prediction')],
                 className='column'),
        # second column : fig performances categorical
        html.Div(children=[dcc.Graph(figure=fig_perfs, className='graph')] + cat_children,
                 className='column')
        #
        html.Div(children=[dcc.Graph(figure=fig_perfs, className='graph')] + DT,
                 className='column')
    ],
             
        className='row')
]
)


# The callback function will provide one "Output" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result", component_property="children"),
              # The values corresponding to sliders and dropdowns of respectively numerical and categorical features
              [Input('{}-dropdown'.format(var), 'value') for var in nums + cats])
# The input variable are set in the same order as the callback Inputs
def update_prediction(*X):
    # get the data input and map it to the correponding feature names
    payload = dict(zip(nums + cats, X))
    # create one line dataframe
    frame_X = pd.DataFrame(payload, index=[0])
    # pass it through the pre-fitted transformer
    X_processed = full_pipeline.transform(frame_X)

    prediction = sk_best.predict_proba(X_processed)[0]

    # And retuned to the Output of the callback function
    return " {}% No , {}% Yes".format("%.2f" % (prediction[0] * 100),
                                      "%.2f" % (prediction[1] * 100))


server = app.server
