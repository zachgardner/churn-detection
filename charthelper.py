from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import datetime
import matplotlib.pyplot as plt



def displayDifferences(series, title, chartType, threshold):
    """
    inputs:
      - series: a list of supervised learners
      - title: a list of dictionaries of the statistic results from 'train_predict()'
      - chartType: The score for the naive predictor
      - threshold: The score for the naive predictor
    """
    #find features with scaled absolute values to test for significance above threshold value 
    series = series[abs(series) > threshold].sort_values()
    #plot the series of features with using the type of chart specific, and title provided. 
    series.plot(title=title, kind=chartType, grid='true', ylim=[-.4,.4])
    
    
def displayAccuracies(y_test, predictions, best_predictions, beta, algorithm, time):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - y_test: the churn result for each of the records in the testing data
      - predictions: the results of the predictions from the default model used
      - best_predictions: the results of the predictions from the optimized model use
      - beta: the measure for beta used in the FScore calculation
      - algorithm: the name of the algorithm used for display purposes
      - time: the amount of time to train the model using grid search for display purposes.,
    """
    #calculate  the accuracy score for the default model
    accuracy_default = accuracy_score(y_test, predictions)
    #calculate the fbeta score for the default model 
    fscore_default = fbeta_score(y_test, predictions, beta = beta)
    #calculate the accuracy score for the optimized model     
    accuracy_optimized = accuracy_score(y_test, best_predictions)
    #calculate the fbeta score for the optimized model     
    fscore_optimized = fbeta_score(y_test, best_predictions, beta = beta)
    #assign the default scores to a series     
    default_scores = [accuracy_default, fscore_default]
    #assign the optimized scores to a series     
    optimized_scores = [accuracy_optimized, fscore_optimized]
    #rename the index for display purposes     
    index = ["Accuracy Score", "FBeta Score"]
    #assemble the optimized and default series' into a dataframe
    df = pd.DataFrame({'Default': default_scores ,'Optimized': optimized_scores}, index=index)
    #plot the results of the model performance in a bar chart.
    df.plot(kind='bar', title=algorithm + " Performance")    
    #print confusion matrix details
    print('Default Confusion Matrix : \n' + str(confusion_matrix(y_test,predictions)))
    print('\nOptimized Confusion Matrix : \n' + str(confusion_matrix(y_test,best_predictions)))
    #print the statistics of the overall experiment
    print("\n" + algorithm +  " Model\n------")
    print("\nUnoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_default))
    print("F-score on testing data: {:.4f}".format(fscore_default))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_optimized))
    print("Final F-score on the testing data: {:.4f}".format(fscore_optimized))
    print("\nPerformance and Improvement\n------")
    print("Time Taken: " + str(datetime.timedelta(seconds=time)))
    print("Accuracy Score Improvement: {:.4f}".format(accuracy_optimized - accuracy_default))
    print("F Score Improvement: {:.4f}\n".format(fscore_optimized - fscore_default))


def displayGridScores(grid_fit, title):
    print("Best parameters set found on development set:")
    print()
    print(grid_fit.best_params_)
    print()

    #get scores from the grid fit ovject
    means = grid_fit.cv_results_['mean_test_score']
    stds = grid_fit.cv_results_['std_test_score']
    reportFrame = pd.DataFrame()

    #initialize series to store the results
    trials = []
    meansData = []
    paramsList = []
    i=0

    #loop to access the parameter pairs, and scores
    for mean, std, params in zip(means, stds, grid_fit.cv_results_['params']):    
        trials.append(i)
        meansData.append(mean)
        paramsList.append(str(params))
        i = i + 1
    #assemble the data frame to report on scores
    reportFrame['trials'] = trials
    reportFrame['means'] = means
    reportFrame['params'] = paramsList
    #take top 5 pairs
    top5 = reportFrame.sort_values('means',ascending=False).head(5)
    #plot the top 5 pairs
    plt.barh(top5['params'],top5['means'],label=top5['params'])
    plt.xlabel('F2 Score')
    plt.title(title + ': Top 5 Performing Parameter Pairs')


