# Sports-Analytics
# 1. Introduction
The English Premier League is a football league that is one of the most watched sports in the world, with an estimated
over 4 billion TV viewers and over $3 billion in revenue from TV rights alone. What makes football exciting is the
unpredictable nature of the game. This may be attributed to the fact that in football, only a few goals total are
awarded within a game, sometimes none at all, reducing the amount of data to predict a successful goal. This is
different from games such as American football for example, where teams on average are earning 24 points in a game
and basketball where teams average over 100 points in a game. Additional factors such as corners, penalties, free
kicks, etc. can impact the game heavily (even leading to additional goals), adding randomness to the overall outcome
and making football results difficult to predict. The goal of this project is to determine the outcome of a game in
the Premier League.
In the Premier League, there are 20 teams. Each team plays all 19 opposing teams twice, as a home game and
away game, making 38 games played per team and 380 games total in a season. There are three outcomes for a
team in a game: 1) win (3 points), 2) lose (0 points), 3) draw (1 point). At the end of all 380 games, the teams are
ranked according to the point tally. Teams ranked in the top 4 and top 6 qualify for international leagues that are
more reputable and offer additional revenue. Thus, outside of predicting game outcomes, determining rankings is
also important.
Several groups have attempted to predict the game outcomes before. Timmaraju et al. 2013 previously listed
three covariates as most important (with others as secondary), namely the number of goals, corners, and shots on
target, on a single season dataset to get above 50% accuracy in game predictions. Ulmer and Fernandez 2013 and
as well as Hessels et al. 2018 used a variety of classification techniques such as SVM and Random forest on a larger
dataset to achieve higher accuracy. Our team will use a variety of different classification models, train them on 15
years of data, and add additional cumulative statistics to improve the prediction performance.
## 1.1 Objective
The objective of this project is to predict the outcome of a game given any two teams in the Premier League. Using
the results, predictions for the total points of each team and the rankings of the top 6 teams will be outputted.
# 2 Methodology
## 2.1 Classification to Determine Game Outcomes
The first step in our methodology will be to reduce the number of covariates under investigation using feature
selection. Using the mutual information approach, covariates that do not provide additional information will be
removed.
I will explore two different classification methods and eighth different models to determine predictions of
each game. For classification, the first classification method will be binary: home win (1) or not (0). The second
classification method will use a multiclassifier: home team wins (0), away team wins(1), or draw (2). I plan
to run experiments using eighth different models: SVM, Random Forest, Logistic Regression, Naive Bayes, Neural
Networks, AdaBoost, K-Nearest Neighbor, and QDA. For each model and classification method, the performance
will be measured using precision, recall, F-1 score, and MSE. Based on the predictions of the games for a given
season, the total point values predicted for each team will be outputted. The average of all the models will be used
to then determine the average total points, and the rankings are determined in descending order.
## 2.2 Split for Training and Testing Datasets
Since this is time series data, I will use historical data to predict more current seasons. 16 seasons between the
years 2000-2016 will be used as training data. Seasons between 2016-2021 will be used as testing data. There is
missing data between the years 2019-2020, which is presumed to be due to COVID, so two seasons were removed
from the testing data. The split is shown in Figure 1.

<img width="662" alt="timeline" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/861e69f0-3c33-40f4-b441-cb02516f8c2b">

# 3 Data Collecting and Preprocessing
## 3.1 Data Source
I have 20 years of data from 2000-2021. However, there is missing data between the years 2019-2020, which is
presumed to be due to COVID, so two seasons were removed. This leads to a total number of over 7200 games. The
following features are included for each game: home team, away team, home team score, away team score, full-time
result, offsides, yellow cards, red cards, fouls, free kicks, shots on target, and total shots. Data is sourced from
https://www.football-data.co.uk/englandm.php.

## 3.2 Feature Selection by Mutual Information Theory
<img width="509" alt="figure2" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/80a6d4cc-38c2-4818-bc52-c4a261f24cf0">

Using mutual information regression, it is determined that corners for both the home and away teams are not
needed in the analysis. These two are removed. Cumulative season statistics are computed from the original features
in order to account for total team statistics as the season progresses. The cumulative number of goals scored, goals
conceded, point accumulated, yellow cards, red cards, fouls, goal difference between teams, conceded goal difference
between teams, and the odds ratio of shots on target are all computed. The cumulative statistics account for all the
information known before the start of each match, and are used as a substitute for the individual match statistics that
occur real-time. Figure 2 shows the resulting correlations between features. Even though the calculated cumulative
statistics correlate to a few of the other features, I keep them in the analysis since through mutual information, we
determined that they provide an information gain in the model. Therefore, the added model complexity is beneficial
to the predictive power of the model. The meanings of each feature is found in the appendix.

# 4 Evaluation and Final Results
## 4.1 Home Win or Not: Binary Classification
The first classification method will be binary: home win (1) or not (0). Eighth different models are used for classification:
SVM, Random Forest, Logistic Regression, Naive Bayes, Neural Networks, AdaBoost, K-Nearest Neighbor,
and QDA. For each model and classification method, the performance will be measured using precision, recall, F-1
score, and MSE. In this model, the points earned by each team cannot be determined. However, determining whether
the home team will win or not is still an important question in a match. An ROC curve is will characterize the
performance of the binary classifiers.
## 4.2 Home Win or Not: Error Metrics and Model Performances
For binary classification, the best performing models were Random Forest, Logistic Regression, AdaBoost and SVM.
In general all of these models have a testing precision, recall, and F1-score between 0.60-0.70 (Table 1). The weighted
averages computed from the confusion matrices show an accuracy near 0.65 (Figure 3). In particular, the Logistic
and SVM models are the best with an AUC of 0.70, followed by AdaBoost. Random Forest suffers from overfitting
on the training data, so it has near perfect performance statistics on training but only 0.63 AUC on the testing data
(Figure 4). Logistic regression and SVM also have the smallest MSE, with only 0.34 and 0.33 respectively (Table 1).
<img width="786" alt="figure3" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/3d909192-7110-4998-aa98-8965558461bc">

<img width="548" alt="figure4" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/45ab5c27-cf1a-4c01-85b5-e83b77102155">

<img width="293" alt="table1" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/90d70a46-d413-44ad-b46d-57debc8bb5c4">

## 4.3 Home Win, Away Win or Draw: Multiclassification
The second classification method will be a multiclassifier: home win (0), away win (1) or draw (2). The same
eighth models are used for classification: SVM, Random Forest, Logistic Regression, Naive Bayes, Neural Networks,
AdaBoost, K-Nearest Neighbor, and QDA. For each model and classification method, the performance will be
measured using precision, recall, F-1 score, and MSE. In this model, the points for each team can be determined
based on the match outcome. Thus, the total points for each team will be predicted and the performance of the total
point prediction will be measured with MSE.
## 4.4 Home Win, Away Win or Draw: Error Metrics and Model Performances
For multiclassification, the best performing models were still Random Forest, Logistic Regression, AdaBoost and
SVM. In particular, SVM and Logistic Regression performed the best once again. Since you cannot create an ROC
Cure for multiclassifiers, I use a One-to-One ROC AUC score for each. When doing so, Logistic, Random Forest,
and SVM recieve an AUC of 0.64, 0.63 and 0.62 (Table 2). The confusion matrices show that these models performed
particularly well in determining the Home wins (Figure 5), which falls in line with the idea that the binary classifier
helps to simplify the problem. Random Forest again suffers from overfitting on the training data with an AUC of
1.00 (Table 2) despite lower testing performance.

<img width="780" alt="figure5" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/51eb8c11-102d-4f12-8aef-10d76bf04fcf">

<img width="405" alt="table2" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/9b0c2e90-3930-4a28-acb4-618ddb456bb0">

## 4.5 Home Win, Away Win or Draw: Predict Total Points per Team
From multiclassification models, I can compute the total points predicted for each team. Recall that a win is 3
points, draw is 1 point, and loss is 0 points. The MSE for each model based on the total point prediction are found
in Table 3. The MSE agrees with the match statistics in Table 2 that SVM and AdaBoost perform particularly
well. In Figure 6, I can see the correlation between the predicted total point values and the true point values that
occurred during the years 2016-2021. There is a positive correlation with a moderate scatter in agreement with the
MSE values. Since different models have different strengths, for example some are better at predicting home games
and others are good at predicting away wins, I will use the average results for all the models to determine the point
count and ultimately predict the rankings.

<img width="594" alt="table3" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/6147bd8b-5c78-4503-9033-95ebeb5fd1c6">

<img width="446" alt="figure6" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/a90896b7-1244-4f47-95cb-658f3b7f0108">

# 5 Prediction of Rankings
Tables 4-6 show the predicted rankings based on the average result of all the multiclassification models. Ranking
in the top 6 is important in the Premier League because it qualifies teams for international leagues around Europe.
This increases revenue, exposure, and prestige. The top 4 teams enter the Champions League and the teams rank 5-6
enter the Europa League. Although I do not predict all of the rankings in order correctly, in general, our models
were able to predict the top 4 teams accurately and the top 6 six teams as well. The only exception is not correctly
prediction West Ham as 6th place in the Season 2020-2021 (Table 6). This shows that although the game-by-game 
predictions may be challenging to predict, the relative strength of the teams compared to one another can still be
estimate

<img width="512" alt="table46" src="https://github.com/macy-mora-antoinette/Sports-Analytics/assets/112992304/e454e700-8867-442f-80d3-dc3b04c34f62">

# 6 Conclusion
In this project the objective was to predict the outcome of a an English football game given any two teams in the
Premier League. Using the match results, predictions for the total points of each team and therefore the rankings of
the top 6 teams were outputted. I initially used binary classification as a simplified version for the problem. We
followed up with multiclassification. Interestingly, the same models performed well for both and the same models
performed relatively poorly for both. Despite that, in general all models showed a positive correlation between
their prediction point values and the true point values. In the end, I took the average total point value from all
the models. This approach was able to successfully determine the top 6 teams that would be able to qualify for
international leagues around Europe. I were also able to show that although the game-by-game predictions may
be challenging to predict, the relative strength of the teams compared to one another can still be estimated

# 7. References
1. Ulmer, B., Fernandez, M., and Peterson, M. (2013). Predicting Soccer Match Results in the Premier League
(Doctoral dissertation, Doctoral dissertation, Ph. D. dissertation, Stanford)
2. Hessels, J. (2018). Improving the prediction of soccer match results by means of Machine Learning. (Masters
thesis, Tilburg University)
3. A. S. Timmaraju, A. Palnitkar, and V. Khanna, Game ON! Predicting English Premier League Match Outcomes,
2013.

# 8. Appendix
In the data, each observation is a game. Features include:
Date = Date of game
HomeTeam
AwayTeam
TFHG = Full Time Home Team Goals
FTAG = Full Time Away Team Goals
FTR = Full Time Results (Home, Away, Draw)
HS = Total Home shots
AS = Total Away shots
HST = Total Home shots on target
AST = Total Away shots on target
HF = Total Home Fouls
AF = Total Away Fouls
HC = Total Home Corners
AC = Total Away Corners
HY = Total Home Yellow cards
AY = Total Away Yellow cards
HR = Total Home Red cards
AR = Total Away Red cards
HGS = Total Home Team Goals Scored
AGS = Total Away Team Goals Scored
HGC = Total Home Team Goals Conceded
AGC = Total Away Team Goals Conceded
HP = Total Home Team Points
AP = Total Away Team Points
HOR = Odds Ratio of Home Team shot on target based on previous shots
AOR = Odds Ratio of Away Team shot on target based on previous shots
Goal Diff = Total Goal difference between teams
Conc Diff = Total Conceded goal difference between teams
