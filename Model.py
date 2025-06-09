import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz

afval = pd.read_csv('voorbeeldAfval.csv')

# Instead of detected_object being a class. I made their classes columns with integer values. Next step is to group them per time interval. 
afval_encoded = pd.get_dummies(afval, columns=['detected_object'], dtype=int)

# Convert timestamp to datetime and extract date
afval_encoded['timestamp'] = pd.to_datetime(afval_encoded['timestamp'], format='%d-%m-%Y %H:%M')
afval_encoded['date'] = afval_encoded['timestamp'].dt.date

# Grouping my data per day
daily_counts = afval_encoded.groupby('date').sum(numeric_only=True).reset_index()

# Convert 'date' back to datetime to extract day_of_week and month features
daily_counts['date'] = pd.to_datetime(daily_counts['date'])
daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek  
daily_counts['month'] = daily_counts['date'].dt.month

# print(daily_counts)
features = ['detected_object_glass', 'detected_object_metal', 'detected_object_organic', 'detected_object_paper', 'detected_object_plastic']


x = daily_counts[['day_of_week', 'month']] # Our training features
y = daily_counts[features]  # Our taget variable

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

dt = DecisionTreeRegressor(max_depth=2)
dt.fit(x_train, y_train)

#---------SCHOOL FUNCTIONS-----------
def plot_tree_regression(model, features):
    # Generate plot data
    dot_data = tree.export_graphviz(model, out_file=None, 
                          feature_names=features,  
                          filled=True, rounded=True,  
                          special_characters=True)  

    # Turn into graph using graphviz
    graph = graphviz.Source(dot_data)  

    # Write out a pdf
    graph.render("decision_tree")

    # Display in the notebook
    return graph 

def calculate_rmse(predictions, actuals):
    if(len(predictions) != len(actuals)):
        raise Exception("The amount of predictions did not equal the amount of actuals")
    
    return (((predictions - actuals) ** 2).sum() / len(actuals)) ** (1/2)
#---------SCHOOL FUNCTIONS-----------

plot_tree_regression(dt, ['day_of_week', 'month'])

predict_train = dt.predict(x_train)
predict_test = dt.predict(x_test)

rmse_train = calculate_rmse(predict_train, y_train.values)
rmse_test = calculate_rmse(predict_test, y_test.values)

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")