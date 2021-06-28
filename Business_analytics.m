clc; clear all;

%load the train dataset
airline_train = readtable('train.csv','ReadVariableNames',true);
airline_train = airline_train(:,3:25);
airline_train = table2dataset(airline_train);
names = airline_train.Properties.VarNames;

%load the test dataset
airline_test = readtable('test.csv','ReadVariableNames',true);
airline_test = airline_test(:,3:25);
airline_test = table2dataset(airline_test);

% Remove unnecessary double quotes from certain attributes
airline_train = datasetfun(@removequotes,airline_train,'DatasetOutput',true);
airline_test = datasetfun(@removequotes,airline_test,'DatasetOutput',true);

% Data Exploration

% Convert all the categorical variables into nominal arrays for train set
[nrows, ncols] = size(airline_train);
category = false(1,ncols);
for i = 1:ncols
    if isa(airline_train.(names{i}),'cell') || isa(airline_train.(names{i}),'nominal')
        category(i) = true;
        airline_train.(names{i}) = nominal(airline_train.(names{i}));
    end
end

% Convert all the categorical variables into nominal arrays for test set
[nrows, ncols] = size(airline_test);
category = false(1,ncols);
for i = 1:ncols
    if isa(airline_test.(names{i}),'cell') || isa(airline_test.(names{i}),'nominal')
        category(i) = true;
        airline_test.(names{i}) = nominal(airline_test.(names{i}));
    end
end

% Logical array keeping track of categorical attributes
catPred = category(1:end-1); 
% Set the random number seed to make the results repeatable in this script
rng('default');

%Missing values 
[row1, col1] = find(ismissing(airline_train));
size(row1); % 310 missing values in column 22

% delete all rows with missing values
airline_train = rmmissing(airline_train);

[row2, col2] = find(ismissing(airline_test));
size(row2); %83 missing values in column 22

% delete all rows with missing values
airline_test = rmmissing(airline_test);


% Visualize Data


%Outliers

outliers = find(airline_train.ArrivalDelayInMinutes>200); 
size(outliers) %807 values
airline_train(airline_train.ArrivalDelayInMinutes>200,:) = [];


% Prepare the Data: Response and Predictors

% Training Set
X_train = double(airline_train(:,1:end-1));

% Response
Y_train = airline_train.satisfaction;
disp('Customer Satisfaction')
tabulate(Y_train);

%Test Set
X_test = double(airline_test(:,1:end-1));

Y_test = airline_test.satisfaction;
disp('Customer Satisfaction')
tabulate(Y_test);

data = [X_train, double(Y_train)-1];
%Find the correlation coefficients
R = corrcoef(data); 
[out, idx] = sort(R(:,end),'descend')
results = [idx out];
%As it can be seen, Online Boarding has the highest positive correlation 
%with satisfaction. Moreover, Type of Travel and Class have the highest negative correlation
%with satisfaction.
%Since the correlation between .00-.19 is considered to be “very weak”, 
%such features as Gender, Ease of Online Booking, Age, Gate Location, 
%Departure/Arrival Time Convenient, Arrival Delay in Minutes,
%Departure Delay in Minutes, and Customer Type can be removed. (1,9,3,10,8,22,21,2)

% Training Set
X_train = double(airline_train(:,[4:7 11:20]));

% Response
Y_train = airline_train.satisfaction;
disp('Customer Satisfaction')
tabulate(Y_train);

%Test Set
X_test = double(airline_test(:,[4:7 11:20]));

Y_test = airline_test.satisfaction;
disp('Customer Satisfaction')
tabulate(Y_test);

% Logical array keeping track of categorical attributes
catPred = category([4:7 11:20]); 
%% Logistic Regression

% Train the classifier
glm = GeneralizedLinearModel.fit(X_train,double(Y_train)-1,'linear','Distribution','binomial','link','logit','CategoricalVars',catPred);


% Make a prediction for the test set
Y_glm = glm.predict(X_test);
Y_glm = round(Y_glm);

% Compute the confusion matrix
C_glm = confusionmat(double(Y_test)-1,Y_glm);

% C_glm =
% 
%        12823        1705
%         2054        9311
        
% Examine the confusion matrix for each class as a percentage of the true class
C_glmperc = bsxfun(@rdivide,C_glm,sum(C_glm,2)) * 100;

% C_glmperc =
% 
%        88.264       11.736
%        18.073       81.927
       
%%Classification Tree
tic
% Train the classifier
t = ClassificationTree.fit(X_train,Y_train,'CategoricalPredictors',catPred);
toc

% Make a prediction for the test set
Y_t = t.predict(X_test);
tabulate(Y_t);

% Compute the confusion matrix
C_t = confusionmat(Y_test,Y_t);

% C_t =
% 
%        13733         795
%          869       10496
         
% Examine the confusion matrix for each class as a percentage of the true class
C_tperc = bsxfun(@rdivide,C_t,sum(C_t,2)) * 100;
    
% C_tperc =
% 
%        94.528       5.4722
%        7.6463       92.354

%%Support Vector Machines
% Train the classifier
svmStruct = fitcsvm(X_train,Y_train,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');

% Make a prediction for the test set
Y_svm = predict(svmStruct,X_test);
C_svm = confusionmat(Y_test,Y_svm);

% C_svm =
% 
%        14096         432
%          831       10534
         
tabulate(Y_svm);

% Examine the confusion matrix for each class as a percentage of the true class
C_svmperc = bsxfun(@rdivide,C_svm,sum(C_svm,2)) * 100;

% C_svmperc =
% 
%        97.026       2.9736
%        7.3119       92.688
    
% Compare Results

Cmat = [C_svmperc C_tperc C_glmperc];
labels = {'Support VM ', 'Decision Trees ', 'Logistic Regression '};

comparisonPlot(Cmat,labels)

%%ROC Curve
%for Logistic Regression
[xxtest1,yytest1,~,auctest1] = perfcurve(Y_test, Y_glm,'satisfied');
auctest1 %0.8510

%For Classification Tree
[xxtest2,yytest2,~,auctest2] = perfcurve(Y_test, double(Y_t)-1,'satisfied');
auctest2 %0.93441

%For SVM
[xxtest3,yytest3,~,auctest3] = perfcurve(Y_test, double(Y_svm)-1,'satisfied');
auctest3 %0.94857


% Plot the new ROC curve
figure;
plot(xxtest1,yytest1, 'b','LineWidth', 2); hold on
plot(xxtest2,yytest2,'g','LineWidth', 2);
plot(xxtest3,yytest3,'r','LineWidth', 2);
xlabel('False positive rate','FontSize', 14); ylabel('True positive rate','FontSize', 14);
title('ROC Curve with test data','FontSize',15);
legend(['Logistic Regression, AUC = ',num2str(auctest1)],['Classification Tree, AUC = ',num2str(auctest2)],['SVM, AUC = ',num2str(auctest3)],'FontSize',14,'Location','southeast'); hold off
