# Air-Pressure-System-Failures-in-Scania-Trucks
Air pressure system failures in Scania trucks, data from UCI ML via Kaggle

See Summary pdf slide for results.

Air Pressure System Failures in Scania Trucks
Data downloaded from Kaggle (UCI ML). https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set License is GPL 2
Data is also available from UCI Machine Learning Repository and is published under the GNU General Public License.

This dataset and challenge was a competition at the 15th Intelligent Data Analysis (IDA 2016) conference. Challenge is to predict failures and minimize cost function of taking trucks off the road.

Cost_1 is 10 for a false positive (cost of unnecessary system inspection) and Cost_2 is 500 for a false negative (cost of a truck missing required service)

From the original competition description:
 The total cost of a prediction model the sum of "Cost_1" 
 multiplied by the number of Instances with type 1 failure 
 and "Cost_2" with the number of instances with type 2 failure, 
 resulting in a "Total_cost".

 In this case Cost_1 refers to the cost that an unnessecary 
 check needs to be done by an mechanic at an workshop, while 
 Cost_2 refer to the cost of missing a faulty truck, 
 which may cause a breakdown.

 Total_cost = Cost_1*No_Instances + Cost_2*No_Instances
 
 The Notebook Scania_Truck_Air_System_Inspection_Prediction loads the data (training & test files) and:
    
   Changes feature values in the import that are object fields to numeric
    
   Ranks features via a Spearman correlation and drops low correlation features
    
   Finds feature importance via an initial Random Forest fit on a reduced set without NaN's, and drops low importance features
    
   Finds a best fit using RandomizedSearch on the reduced set (~95% of samples) 
    
   Fills any NaN's with the mode value for each feature in both Training and Test sources
    
   Fits a RandomForestClassifier on the final NaN filled training set using the hyperparametes from the RandomizedSearch
    
   Predicts and costs the false negatives plus false positives on the Test set 
