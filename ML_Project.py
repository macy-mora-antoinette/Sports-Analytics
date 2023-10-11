#!/usr/bin/env python
# coding: utf-8

# # Predicting Results for the English Premier League
# ### Authors: Adrien Antoinette and Macy Castaneda <br>
# Data sourced from Kaggle [here](https://www.kaggle.com/saife245/football-match-prediction/data)

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime as dt
import itertools
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.metrics import roc_auc_score
pd.options.mode.chained_assignment = None
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Data as Pandas Dataframe
# 
# Each observation is a game. Features include:
# 
# 
# * Date = Date of game
# * HomeTeam
# * AwayTeam
# * TFHG = Full Time Home Team Goals
# * FTAG = Full Time Away Team Goals
# * FTR = Full Time Results (Home, Away, Draw)
# * HS = Home shots
# * AS = Away shots
# * HST = Home shots on target
# * AST = Away shots on target
# * HF = Home Fouls
# * AF = Away Fouls
# * HC = Home Corners
# * AC = Away Corners
# * HY = Home Yellow cards
# * AY = Away Yellow cards
# * HR = Home Red cards
# * AR = Away Red cards

# In[2]:


#directory with data 
folder='./EPL/' 

#columns selected: goals, shots, target shots, fouls, corners, yellow and red cards for each team
cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS','HST','AST','HF','AF','HC','AC', 'HY','AY','HR','AR']

#store data in temporary list
ls = []
for i in range(0,21):
    filename = '20'+str(i).zfill(2)+'-'+str(i+1).zfill(2)+'.csv'          
    ls.append(pd.read_csv(folder +filename))  
    ls[i] = ls[i][cols]
    ls[i]['season'] = i #add season as a column

#turn list into a df
df = pd.concat(ls, ignore_index=True)
df.Date =pd.to_datetime(df.Date, dayfirst=True)

#Show df
print("dataframe shape:", df.shape)
#df.head()


# ## Feature Selection with Mutual Information
# 
# Feature selection says to remove yellow cards for both teams

# In[3]:


#mutual info to find factors
X_all= df.copy()
X_all = X_all.drop(['FTR', 'Date','HomeTeam','AwayTeam', 'season'],1).astype('float64')
y_all = df['FTR'].replace(to_replace ={'H': 0, 'D': 2, 'A': 1})
mi = mutual_info_regression(X_all, y_all, random_state = 7)
mi /= np.max(mi)
print(df.columns[np.where(mi == 0)[0]])
#use info from mutual information to filter 
df = df.drop(df.columns[np.where(mi == 0)[0]], axis='columns')

#df.head()


# ## Compute Cummulative Match Statistics
# 
# These are the cummulative stats at the start of each match. That is, they are known before the match begins.
# * HS = Total Home shots
# * AS = Total Away shots
# * HST = Total Home shots on target
# * AST = Total Away shots on target
# * HF = Total Home Fouls
# * AF = Total Away Fouls
# * HY = Total Home Yellow cards
# * AY = Total Away Yellow cards
# * HR = Total Home Red cards
# * AR = Total Away Red cards
# * HGS = Total Home Team Goals Scored
# * AGS = Total Away Team Goals Scored
# * HGC = Total Home Team Goals Conceded
# * AGC = Total Away Team Goals Conceded
# * HP = Total Home Team Points 
# * AP = Total Away Team Points
# * HOR = Odds Ratio of Home Team shot on target based on previous shots
# * AOR = Odds Ratio of Away Team shot on target based on previous shots
# 

# In[4]:


#Add new columns to df
df = pd.concat([df, pd.DataFrame(columns = ['HGS', 'AGS', 'HGC', 'AGC', 'HP', 'AP', 'HOR', 'AOR', 'HPT', 'APT'])])

#iterate over list of all seasons
seasons = df['season'].unique()
for season in seasons:
    df_season = df.loc[df.season == season]
    df_season['points'] =  "" #will be used to store goals and compute stats
    
    #iterate over list of all teams in that season
    teams = df_season['HomeTeam'].unique() 
    for team in teams:
              
        #account for home games
        home = df_season.loc[df_season['HomeTeam'] == team]  
        home['scored'] = home['FTHG']
        home['conceded'] = home['FTAG']
        home['shots'] = home['HS']
        home['targets'] = home['HST']
        home.points[home.FTHG > home.FTAG] = 3 #win
        home.points[home.FTHG == home.FTAG] = 1 #draw
        home.points[home.FTHG < home.FTAG] = 0 #lose
        home['fouls'] = home['HF']
        home['yellowflags'] = home['HY']
        home['redflags'] = home['HR']

        #account for away games
        away = df_season.loc[df_season['AwayTeam'] == team]
        away['scored'] = away['FTAG']
        away['conceded'] = away['FTHG']
        away['shots'] = away['AS']
        away['targets'] = away['AST']
        away.points[away.FTHG < away.FTAG] = 3 #win
        away.points[away.FTHG == away.FTAG] = 1 #draw
        away.points[away.FTHG > away.FTAG] = 0 #lose
        away['fouls'] = away['HF']
        away['yellowflags'] = away['HY']
        away['redflags'] = away['HR']
        
        #combined home and away results 
        df_team = pd.concat([home, away])
        df_team.sort_values(by = 'Date', inplace = True)

        #compute cummulative stats
        #FTHG and FTAG
        df_team['cum_scored'] = df_team['scored'].cumsum().shift(1).fillna(0)
        df_team['cum_conceded'] = df_team['conceded'].cumsum().shift(1).fillna(0)
        #HST/AST and HS/AS
        df_team['cum_shots'] = df_team['shots'].cumsum().shift(1)
        df_team['cum_targets'] = df_team['targets'].cumsum().shift(1)
        df_team['probability'] = df_team['cum_targets']/df_team['cum_shots'].values
        df_team['odds_ratio'] = (df_team['probability']/(1-df_team['probability'].values)).fillna(0)
        #Points
        df_team['cum_points'] = df_team['points'].cumsum().shift(1).fillna(0)
        df_team['cum_points2'] = df_team['points'].cumsum()
        #Fouls/YellowFlags/RedFlags
        df_team['cum_fouls'] = df_team['fouls'].cumsum().shift(1)
        df_team['cum_yellowflags'] = df_team['yellowflags'].cumsum().shift(1)
        df_team['cum_redflags'] = df_team['redflags'].cumsum().shift(1)
        
        #assign stats to proper home or away columns in df
        #shots and targets
        df.HS[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_shots[df_team.HomeTeam == team]
        df.AS[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_shots[df_team.AwayTeam == team] 
        df.HST[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_targets[df_team.HomeTeam == team]
        df.AST[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_targets[df_team.AwayTeam == team] 
        df.HOR[(df_team.HomeTeam == team) & (df.season == season)] = df_team.odds_ratio[df_team.HomeTeam == team]
        df.AOR[(df_team.AwayTeam == team) & (df.season == season)] = df_team.odds_ratio[df_team.AwayTeam == team]
        #scores and conceded
        df.HGS[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_scored[df_team.HomeTeam == team]
        df.AGS[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_scored[df_team.AwayTeam == team] 
        df.HGC[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_conceded[df_team.HomeTeam == team]
        df.AGC[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_conceded[df_team.AwayTeam == team]
        #Points
        df.HP[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_points[df_team.HomeTeam == team]
        df.AP[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_points[df_team.AwayTeam == team]       
        df.HPT[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_points2[df_team.HomeTeam == team]
        df.APT[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_points2[df_team.AwayTeam == team]
        #Fouls/YellowFlags/RedFlags
        df.HF[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_fouls[df_team.HomeTeam == team]
        df.AF[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_fouls[df_team.AwayTeam == team] 
        df.HY[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_yellowflags[df_team.HomeTeam == team]
        df.AY[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_yellowflags[df_team.AwayTeam == team]
        df.HR[(df_team.HomeTeam == team) & (df.season == season)] = df_team.cum_redflags[df_team.HomeTeam == team]
        df.AR[(df_team.AwayTeam == team) & (df.season == season)] = df_team.cum_redflags[df_team.AwayTeam == team]

#Replace a 100% target/shot (odds ratio = Infinity) with a very high value 
df.replace(to_replace = np.inf, value = 1000, inplace = True)
df.replace(to_replace = np.nan, value = 0, inplace = True)
df = df[(df.season != 18) & (df.season != 19)]
df['Goal Diff'] = df['HGS'] - df['AGS']
df['Conc Diff'] = df['HGC'] - df['AGC']
df = df.drop(['FTHG', 'FTAG'],1)
df


# ## Correlation Matrix of Features

# In[5]:


plt.figure(figsize=(20,10)) 
sns.heatmap(df.drop(['HPT', 'APT'], axis = 1).corr(),
                cmap="GnBu",  # Choose a squential colormap
                annot_kws={'fontsize':11},  # Reduce size of label to fit
                fmt='',
                square=True,     # Force square cells
                linewidth=1,  # Add gridlines
                linecolor="white"# Adjust gridline color
               )


# # <u>Model for Binary Classifier: Home Win or Not</u>
# 
# Home Win is (1), otherwise (0)<br>
# 
# Training data is seasons 2000-2016 <br>
# Testing data is seasons 2016-2017, 2017-2018 and 2020-2021

# In[6]:


#Create binary label for home wins
train1_df = df.drop(['HPT', 'APT'], axis = 1).copy()
train1_df.FTR = train1_df.FTR.replace(to_replace ={'H': 1, 'D': 0, 'A': 0})

#Separate into feature set and target variable
datscatter = train1_df.copy()
X_all = datscatter.drop(['FTR', 'Date','HomeTeam','AwayTeam', 'season'],1).astype('float64')
y_all = datscatter['FTR']

#Split into testing and training set
train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
test = [16, 17, 20]
X_train, y_train = X_all[df.season.isin(train)], y_all[df.season.isin(train)]
X_test, y_test = X_all[df.season.isin(test)], y_all[df.season.isin(test)]

# Dictionaries used to store MSE, ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
mse = dict()


# ## Classifiers
# We try 8 different classifiers
# * Logistic Regression
# * SVM Kernel
# * Random Forest
# * KNN
# * QDA
# * AdaBoost
# * Neural Networks
# * Naive Bayes

# In[7]:


#Create first four models
classifiers1 = [RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1),
              KNeighborsClassifier(3),
              MLPClassifier(alpha=1, max_iter=1000),
              GaussianNB()]

names1 = ["Random Forest", "KNN", "Neural Network", "Naive Bayes"]

for classifier, name in zip(classifiers1, names1):
    #Model Training
    classifier.fit(X_train, y_train)
    Y_pred = classifier.predict(X_test)#predicting result
    Y_predtrain = classifier.predict(X_train)#predicting result

    #Create training and testing dictionaries
    mse[name] = dict.fromkeys(['train', 'test'])
    #Compute MSE and micro-average ROC curve and ROC area
    mse[name]['train'] = mean_squared_error(y_train, Y_predtrain)
    mse[name]['test'] = mean_squared_error(y_test, Y_pred)
    fpr[name], tpr[name], _ = roc_curve(y_test.ravel(), Y_pred.ravel())
    roc_auc[name] = auc(fpr[name], tpr[name])

    # Print MSE and Confusion Matrix
    print(name)    
    print('train')
    print(classification_report(y_train, Y_predtrain))
    print('test')
    print(classification_report(y_test, Y_pred))

    cm = confusion_matrix(y_test, Y_pred)
    plt.figure()
    sns.heatmap(cm,
                    cmap="ocean",  # Choose a squential colormap
                    annot_kws={'fontsize':11},  # Reduce size of label to fit
                    fmt='d',
                    annot = True, # Interpret labels as strings
                    square=True,     # Force square cells
                    linewidth=0.1,  # Add gridlines
                    linecolor="k" # Adjust gridline color
                   )


# In[8]:


#Create second four models
classifiers2 = [LogisticRegression(random_state = 0),
              SVC(kernel = 'rbf',random_state = 0),
              QuadraticDiscriminantAnalysis(),
              AdaBoostClassifier()]

names2 = ["Logistic", "SVM", "QDA", "AdaBoost"]
for classifier, name in zip(classifiers2, names2):
    #Model Training
    classifier.fit(X_train, y_train)
    Y_pred = classifier.predict(X_test)
    Y_predtrain = classifier.predict(X_train)#predicting result
    y_score = classifier.fit(X_train, y_train).decision_function(X_test) #when finding home win 
    y_scoretrain = classifier.fit(X_train, y_train).decision_function(X_train)
    
    #Create training and testing dictionaries
    mse[name] = dict.fromkeys(['train', 'test'])
    #Compute MSE and micro-average ROC curve and ROC area
    mse[name]['train'] = mean_squared_error(y_train, Y_predtrain)
    mse[name]['test'] = mean_squared_error(y_test, Y_pred)
    fpr[name], tpr[name], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc[name] = auc(fpr[name], tpr[name])

    # Print MSE and Confusion Matrix
    print(name)    
    print('train')
    print(classification_report(y_train, Y_predtrain))
    print('test')
    print(classification_report(y_test, Y_pred))
    
    cm = confusion_matrix(y_test, Y_pred)
    plt.figure()
    sns.heatmap(cm,
                    cmap="ocean",  # Choose a squential colormap
                    annot_kws={'fontsize':11},  # Reduce size of label to fit
                    fmt='d',
                    annot = True, # Interpret labels as strings
                    square=True,     # Force square cells
                    linewidth=0.01,  # Add gridlines
                    linecolor="k"# Adjust gridline color
                   )


# In[9]:


mse


# In[10]:


# Plot all ROC curves
plt.figure()
names = ["Random Forest", "KNN", "Neural Network", "Naive Bayes", "Logistic", "SVM", "QDA", "AdaBoost"]
colors = ["green", "lime", "cyan", "blue", "teal", "deepskyblue", "grey", "lightgreen"]
for name, color in zip(names, colors):
    plt.plot(
        fpr[name],
        tpr[name],
        label=name + " (area = {0:0.2f})".format(roc_auc[name]),
        color=color,
        linestyle=":",
        linewidth=4,
    )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve for Binary Class")
plt.legend(loc="lower right")


# # <u>Model for Multiclassifier: Home Win, Away Win, or Draw</u>
# 
# Home Win is (0), Away Win is (1), and Draw is (2)<br>
# 
# Training data is seasons 2000-2016 <br>
# Testing data is seasons 2016-2017, 2017-2018 and 2020-2021

# In[11]:


#Multilabels: home = 0, away = 1, draw = 2
train_df = df.drop(['HPT', 'APT'], axis = 1).copy()
train_df.FTR = train_df.FTR.replace(to_replace ={'H': 0, 'D': 2, 'A': 1})

#Separate into feature set and target variable
datscatter = train_df.copy()
X_all = datscatter.drop(['FTR', 'Date','HomeTeam','AwayTeam', 'season'],1).astype('float64')
y_all = datscatter['FTR']

#Split into testing and training set
train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
test = [16, 17, 20]
X_train2, y_train2 = X_all[df.season.isin(train)], y_all[df.season.isin(train)]
X_test2, y_test2 = X_all[df.season.isin(test)], y_all[df.season.isin(test)]

# Dictionaries used to store ROC curve and ROC area for each class
mse_multi = dict()


# ## Classifiers
# We try 8 different classifiers
# * Logistic Regression
# * SVM Kernel
# * Random Forest
# * KNN
# * QDA
# * AdaBoost
# * Neural Networks
# * Naive Bayes

# In[12]:


#Create all eight models
classifiers = [RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1),
              KNeighborsClassifier(3),
              MLPClassifier(alpha=1, max_iter=1000),
              GaussianNB(),
              LogisticRegression(random_state = 0),
              SVC(kernel = 'rbf',random_state = 0, probability = True),
              QuadraticDiscriminantAnalysis(),
              AdaBoostClassifier()]
    
names = ["Random Forest", "KNN", "Neural Network", "Naive Bayes", "Logistic", "SVM", "QDA", "AdaBoost"]
Y_pred_multi = dict() #store predictions to predict rankings later
for classifier, name in zip(classifiers, names):
    classifier.fit(X_train2, y_train2)
    Y_predtrain2 = classifier.predict(X_train2)#predicting result
    Y_pred_multi[name] = classifier.predict(X_test2)

    # when finding h,w,d
    #TRAINING
    y_probtrain2 = classifier.predict_proba(X_train2)
    macro_roc_auc_ovo_train = roc_auc_score(y_train2, y_probtrain2, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo_train = roc_auc_score(y_train2, y_probtrain2, multi_class="ovo", average="weighted")
    print(name)
    print("train")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} " "(weighted by prevalence)".format(macro_roc_auc_ovo_train, weighted_roc_auc_ovo_train))  
    
    #TESTING
    y_prob2 = classifier.predict_proba(X_test2)
    macro_roc_auc_ovo = roc_auc_score(y_test2, y_prob2, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_test2, y_prob2, multi_class="ovo", average="weighted")
    print("test")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} " "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    
    #Create training and testing dictionaries for MSE
    mse_multi[name] = dict.fromkeys(['train', 'test'])
    #Compute MSE and micro-average ROC curve and ROC area
    mse_multi[name]['train'] = mean_squared_error(y_train2, Y_predtrain2)
    mse_multi[name]['test'] = mean_squared_error(y_test2, Y_pred_multi[name])

    # Print MSE and Confusion Matrix
    print(name)    
    print('train')
    print(classification_report(y_train2, Y_predtrain2))
    print('test')
    print(classification_report(y_test2,Y_pred_multi[name]))

    cm = confusion_matrix(y_test2, Y_pred_multi[name])
    plt.figure()
    sns.heatmap(cm,
                cmap="ocean",  # Choose a squential colormap
                annot_kws={'fontsize':11},  # Reduce size of label to fit
                fmt='d',
                annot = True, # Interpret labels as strings
                square=True,     # Force square cells
                linewidth=0.01,  # Add gridlines
                linecolor="k"# Adjust gridline color
               )
    
print('MSE')
mse_multi


# ## Compute Premier League Point Totals
# 
# ### Points
# For multiclassifier, we can compute the total points for each team and acquire MSE. Compute for the testing dataset seasons 2017-2021

# In[13]:


names = ["Random Forest", "KNN", "Neural Network", "Naive Bayes", "Logistic", "SVM", "QDA", "AdaBoost"]
test = [16, 17, 20]
df2 = df[df.season.isin(test)].reset_index()
df2 = pd.concat([df2, pd.DataFrame.from_dict(Y_pred_multi)], axis = 1)
points = dict()
#iterate over testing seasons
for season in df2['season'].unique():
    df_season = df2.loc[df2.season == season]
    df_season['points'] =  "" #will be used to store goals and compute stats
    df_points = pd.DataFrame(columns = ['Team', 'True Points'] + names ) #store rankings for a given season
    
    #iterate over list of all teams in that season
    teams = df_season['HomeTeam'].unique() 
    df_points['Team'] = teams #store teams in rankings df
    for team in teams:
        #Account for home and away team
        home = df_season.loc[df_season['HomeTeam'] == team]
        away = df_season.loc[df_season['AwayTeam'] == team]  
        
        #Determining True Point Value
        df_team_true = pd.concat([home, away])
        last_game = df_team_true[df_team_true.Date == df_team_true.Date.max()]        
        if list(last_game.iloc[0, 2:4]).index(team) == 0: #If team is HomeTeam on last game then it has index 0 
            df_points['True Points'][df_points.Team == team] = int(last_game['HPT'])
        else: #If team is AwayTeam on last game then it has index 1 
            df_points['True Points'][df_points.Team == team] = int(last_game['APT'])
        
        #Iterate over models
        for model in names:     
            #account for home games
            home.points[home[model] == 0] = 3 #win
            home.points[home[model] == 2] = 1 #draw
            home.points[home[model] == 1] = 0 #lose

            #account for away games
            away.points[away[model] == 1] = 3 #win
            away.points[away[model] == 2] = 1 #draw
            away.points[away[model] == 0] = 0 #lose

            #combined home and away results 
            df_team = pd.concat([home, away])
            df_team.sort_values(by = 'Date', inplace = True)

            #compute cummulative points
            df_points[model][df_points.Team == team] = df_team['points'].cumsum().max()
      
    df_points.sort_values(by = 'True Points', ascending = False, inplace = True, ignore_index = True)
    points[season] = df_points
    
points[20]


# In[14]:


names = ["Random Forest", "KNN", "Neural Network", "Naive Bayes", "Logistic", "SVM", "QDA", "AdaBoost"]
colors = ["purple", "lime", "cyan", "blue", "teal", "deepskyblue", "grey", "gold"]
mse_points = dict()
for key in points:
    mse_points[key] = dict()
    #create a figure for each season tested
    plt.figure()
    min_val = min(points[key].iloc[:,1::].min()) - 5
    max_val = max(points[key].iloc[:,1::].max()) + 5
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel('True Point Total')
    plt.ylabel('Predicted Point Total') 
    plt.title('Multiclass Season 20' + str(int(key)) + '-20' + str(int(key+1)))
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    #plot each model
    x = list(points[key]['True Points']) #x axis is True Points 
    for name, color in zip(names, colors):
        y = list(points[key][name]) #Y axis is model Points
        plt.plot(x, y, marker='o', lw = 0, color=color, label = name)
        mse_points[key][name] = mean_squared_error(x, y)
    plt.legend(loc = 'upper left')
    
mse_points


# In[15]:


season16_avg = pd.concat([points[16].iloc[:,0:2], points[16].iloc[:,2::].mean(axis = 1)], axis = 1)
season16_avg


# In[16]:


season17_avg = pd.concat([points[17].iloc[:,0:2], points[17].iloc[:,2::].mean(axis = 1)], axis = 1)
season17_avg


# In[17]:


season20_avg = pd.concat([points[20].iloc[:,0:2], points[20].iloc[:,2::].mean(axis = 1)], axis = 1)
season20_avg


# In[ ]:




