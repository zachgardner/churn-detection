
# Supervised Learning
## Project: Finding Donors for CharityML

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

Optional 
[Flask]: (http://flask.pocoo.org)
[sklearn2pmml] : https://github.com/jpmml/sklearn2pmml 

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Code

Template code is provided in the `churn_detection_capstone.ipynb` notebook file. You will also be required to use the included `charthelper.py` Python file and the `churn_data.csv` dataset. 
### Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook ./project/churn_detection_capstone.ipynb
```  
or
```bash
jupyter notebook churn_detection_capstone.ipynb
```

This will open the iPython Notebook software and project file in your browser.


To run the Flask API Locally:

open another command line in the project folder location and execute "python ./project/flask-api.py". Enable the api flag in the second to last cell in the notebook. The default code will run the API using AWS Lambda.

### Data

The modified census dataset consists of approximately 7,000 data points, with each datapoint having 21 features. This dataset is a modified version of the dataset published on Kaggle https://www.kaggle.com/blastchar/telco-customer-churn 

**Features**
* **customerID**: Customer ID
* **gender**:Whether the customer is a male or a female
* **SeniorCitizen**: Whether the customer is a senior citizen or not (1, 0)
* **Partner**: Whether the customer has a partner or not (Yes, No)
* **Dependents**: Whether the customer has dependents or not (Yes, No)
* **tenureNumber**: # of months the customer has stayed with the company
* **PhoneService**: Whether the customer has a phone service or not (Yes, No)
* **MultipleLines**: Whether the customer has multiple lines or not (Yes, No, No phone service)
* **InternetService**: Customer’s internet service provider (DSL, Fiber optic, No)
* **OnlineSecurity**: Whether the customer has online security or not (Yes, No, No internet service)
* **OnlineBackup**: Whether the customer has online backup or not (Yes, No, No internet service)
* **DeviceProtection**: Whether the customer has device protection or not (Yes, No, No internet service)
* **TechSupport**: Whether the customer has tech support or not (Yes, No, No internet service)
* **StreamingTV**: Whether the customer has streaming TV or not (Yes, No, No internet service)
* **StreamingMoviesWhether**: the customer has streaming movies or not (Yes, No, No internet service)
* **Contract**: The contract term of the customer (Month-to-month, One year, Two year)
* **PaperlessBilling**: Whether the customer has paperless billing or not (Yes, No)
* **PaymentMethod**: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
* **MonthlyCharges**: The amount charged to the customer monthly
* **TotalCharges**: The total amount charged to the customer
* **Churn**: Whether the customer churned or not (Yes or No)
**Target Variable**
- `Churn`: Yes, No

### Project Contents

* **charthelper.py**: Python library containing user defined functions to display visualizations
* **churn_data.csv**: The primary source data used for training and testing sets.
* **ChurnModel.pmml**: The developed model, extracted for consumption via PMML.
* **model.pkl**: The file holding a persistent object of the predictive model used for training. Obtained using pickle library. 
* **columns.pkl**: The file holding a persistent object of the data frame columns used for training. Obtained using pickle library. 
* **flask-api.py**: Standalone python script used to host, and deploy model of real-time web service.
* **zappa-settings.json**: Zappa deployment properties for sending the code to AWS Lambda 