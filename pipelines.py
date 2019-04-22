

# Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score



#region DATA

# preparation
y = df['y'].values  # .values -> np array (df otherwise)
X = df.drop('y', axis=1).values  # exclude y from regressors
# reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))  # (139, )
print("Dimensions of X before reshaping: {}".format(X.shape))  # (139, )
y = y.reshape(-1, 1)  # sklearn needs certain array-shape!!!!
X = X.reshape(-1, 1)
print("Dimensions of y after reshaping: {}".format(y.shape))  # (139, 1)
print("Dimensions of X after reshaping: {}".format(X.shape))  # (139, 1)

# training data vs test data
# problem: model's ability to generalize
# solution: split data, fit model on X_train and predict on X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42, stratify=y)  # stratification -> labels distr. in training/test data like in original data

#endregion (DATA)





#region CLASSIFICATION
# discrete dependent variable

# kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)  # Model
knn.fit(X_train, y_train)  # Fitting
y_pred = knn.predict(X_test)  # Predicting
print(knn.score(X_train, y_train))  # Accuracy train data
print(knn.score(X_test, y_test))  # Accuracy test data

# model complexity for kNN (-> choose n_neighbors -> hyperparm tuning section)
neighbors = np.arange(1, 9)  # values to consider for n_neighbors
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):  # index, values - pairs of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)  # instantiation
    knn.fit(X_train, y_train)  # Fitting
    train_accuracy[i] = knn.score(X_train, y_train)  # Accuracy train data
    test_accuracy[i] = knn.score(X_test, y_test)  # Accuracy test data
# Generate plot
_ = plt.title('k-NN: Varying Number of Neighbors')
_ = plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
_ = plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
_ = plt.legend()
_ = plt.xlabel('Number of Neighbors')
_ = plt.ylabel('Accuracy')
plt.show()  # choose sweetspot

# logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()  # Model
logreg.fit(X_train, y_train)  # Fitting
y_pred = logreg.predict(X_test)  # Predicting
print(logreg.score(X_train, y_train))  # Accuracy train data
print(logreg.score(X_test, y_test))  # Accuracy test data

# Decision Tree (-> tuning section)

# SVC (-> Imputing in 18)

#endregion (CLASSIFICATION)





#region REGRESSION
# continuous dependent variable

# Linear Regression (OLS)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()  # Model
reg.fit(X_train, y_train)  # Fitting
y_pred = reg.predict(X_test)  # Predicting
print(reg.score(X_train, y_train))  # Accuracy training data (R^2)
print(reg.score(X_test, y_test))  # Accuracy test data (R^2)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # RMSE

# Decision Tree

#endregion (REGRESSION)





#region REGULARIZED REGRESSION
# problem: overfit due to many features and large coeffs.
# solution: alter loss funs and add penalty for large coeffs.

# Lasso (-> first choice for feature selection!)
# adds sum of absolute coeff to loss fun
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=.4, normalize=True)  # Model
lasso.fit(X, y)  # Fitting
lasso_coef = lasso.coef_  # access coeff.
print(lasso_coef)  # print coeff
_ = plt.plot(range(len(df_columns)), lasso_coef)  # plot coeff!
_ = plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
_ = plt.margins(0.02)
plt.show()

# Ridge (-> first choice for regression models!)
# adds sum of squared coeff to loss fun
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
ridge = Ridge(normalize=True)  # Model
# which alpha??
alpha_space = np.logspace(-4, 0, 50)  # array of alphas
ridge_scores = []  # lists to store scores
ridge_scores_std = []
for alpha in alpha_space:  # compute scores over range of alphas
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha  # access Ridge(alpha=)!!!!!
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))
display_plot(ridge_scores, ridge_scores_std)

# ElasticNet() (-> see Tuning section)

#endregion (REGULARIZED REGRESSION)





#region CROSS VALIDATION
# problem 1: performance depends on way the data is split
# problem 2: overfit to (one) sample
# solution: cross validation!
# note: using ALL data for cv not ideal -> train_test_split, cv on train. data
# note: mind the cv vs computing power-tradeoff)

from sklearn.model_selection import cross_val_score
reg = LinearRegression()  # model selection
cv_scores = cross_val_score(reg, X, y, cv=5)  # 5-fold cross-validation scores
print(cv_scores)  # R^2
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))  # R^2

#endregion (CROSS VALIDATION)





#region MODEL ACCURACY MEASURES

# standard accuracy measures
print(model.score(X_train, y_train))  # Accuracy train data (R^2 for cont. var.)
print(model.score(X_test, y_test))  # Accuracy test data

# mean squared error
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)  # Predicting
mse = mean_squared_error(y_test, y_pred)

# confusion matrix, classificaton report (Classification)
# problem: class imbalance leads automatically to high accuracy measures
# solution: confusion matrix, classificaton report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = logreg.predict(X_test)  # Predicting
print(confusion_matrix(y_test, y_pred))  # confusion matrix
print(classification_report(y_test, y_pred))  #  classification report

# ROC-curve (Classification, logreg)
# visual evaluation. true pos.&false pos. rates in fun of threshold
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]  # .predict_proba returns two columns (probas for binary outcomes). we're interested in second column -> [:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  # fpr, tpr, thresholds
_ = plt.plot([0, 1], [0, 1], 'k--')  # Plot ROC curve
_ = plt.plot(fpr, tpr)
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
plt.show()

# AUC score (area under ROC curve), w/ cross validation
# idea: random guesses = .5 -> diagonal line
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
y_pred_prob = logreg.predict_proba(X_test)[:,1]  # predicted probabilities
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))  # AUC score
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')  # AUC cv!
print("AUC scores using 5-fold cross-validation: {}".format(cv_auc))

#endregion (MODEL ACCURACY MEASURES)





#region HYPERPARAMETER TUNING
# problem: how to choose regularization params of model (ridge, lasse: alpha; logreg: C; kNN: n_neighbors; etc.)
# solution: hyperparam tuning
# note: don't use ALL data for tuning -> holdout set!

# GridSearchCV (w/ logreg)
from sklearn.model_selection import GridSearchCV
c_space = np.logspace(-5, 8, 15)  # Setup the hyperparam space for 'C'
param_grid = {'C': c_space}  # Setup the hyperparam grid (one hyperparam)
logreg = LogisticRegression()  # Model (C-default: 1.0)
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)  # Inst. GridSearchCV
logreg_cv.fit(X_train, y_train)  # Fitting on training data
# best hyperparameter
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# GridSearchCV (w/ ElasticNet)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
l1_space = np.linspace(0, 1, 30)  # hyperparam space
param_grid = {'l1_ratio': l1_space}  # hyperparam grid
elastic_net = ElasticNet()  # Model
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)  # Inst. GridSearchCV
gm_cv.fit(X_train, y_train)  # Fitting on traing data
y_pred = gm_cv.predict(X_test)  # Predicting
r2 = gm_cv.score(X_test, y_test)  # Accuracy test (R^2)
mse = mean_squared_error(y_test, y_pred)  # Accuracy test
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

# RandomizedSearchCV (w/ DecisionTreeClassifier)
# problem: large hyperparam spaces, many hyperparams -> GridSearchCV comp. exp.
# solution: fixed number of hyperparam values is sampled
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
# Setup the grid and hyperparam spaces to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()  # Model
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5, n_iter=10)  # Inst. RandomizedSearchCV. n_iter: the number of parameter settings tried
tree_cv.fit(X_train, y_train)  # Fitting on training data
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

#endregion (HYPERPARAMETER TUNING)





#region PREPROCESSING

# Dummies
df_region = pd.get_dummies(df, drop_first=True)  # avoid dummy trap
print(df_region.columns)

# Missing data 1: dropping NaNs
df[df == '?'] = np.nan  # Convert '?' to NaN
print(df.isnull().sum())  # number of NaNs
print(df.shape)  # shape of original df
print(df.dropna().shape)  # shape after dropping NaNs. acceptable?

# Missing data 2: imputing NaNs
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X_imp = imp.transform(X)

# Normalizing/Standardizing Data
# problem: plethora of scales unduly influences model
# solution: normalizing data (standardization (mu=0,sigma2=1), normalization (data in range(-1,1)), subtract min and divide by range (data in range(0,1)))
from sklearn.preprocessing import scale
X_scaled = scale(X)
np.mean(X), np.std(X)  # (8.13, 16.72)
np.mean(X_scaled), np.std(X_scaled)  # (0, 1)

# NLP
from sklearn.feature_extraction.text import CountVectorizer
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'  # token pattern
df.Position_Extra.fillna('', inplace=True)  # Fill missing values
# Instantiate the CountVectorizer
vec_alphanumeric = CountVectorizer(
    token_pattern=TOKENS_ALPHANUMERIC,
    ngram_range=(1, 2))
vec_alphanumeric.fit(df.Position_Extra)  # Fitting
# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])


# Interaction terms

#endregion (PREPROCESSING)




#region HASHING
# increases computational efficiency

from sklearn.feature_extraction.text import HashingVectorizer
# Get text data: text_data
text_data = combine_text_columns(X_train)
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 
# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)
# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())
#endregion (HASHING)





#region CLASSIFICATION PIPELINES
# pipeline: in a repeatable way from raw data to trained model

# Scaling and Tuning in a Classification Pipeline
# pipeline
steps = [('scaler', StandardScaler()),  # scale data
         ('SVM', SVC())]  # Model
pipeline = Pipeline(steps)  # Pipeline
# tuning
parameters = {'SVM__C':[1, 10, 100],  # hyperparam grid
              'SVM__gamma':[0.1, 0.01]}  # 'pipeline-step-name'_ _'param-name'
cv = GridSearchCV(pipeline, param_grid=parameters)  # Inst. GridSearchCV
# holdout-set reasoning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=21)
cv.fit(X_train, y_train)  # Fitting
y_pred = cv.predict(X_test)  # Predicting
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))






#endregion (CLASSIFICATION PIPELINES)





#region REGRESSION PIPELINES
# pipeline: in a repeatable way from raw data to trained model

# NaNs, Scaling and Tuning in a Regression Pipeline
# pipeline
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]  # Model
pipeline = Pipeline(steps)  # pipeline
# tuning
parameters = {'elasticnet__l1_ratio': np.linspace(0,1,30)}  # param grid
gm_cv = GridSearchCV(pipeline, param_grid=parameters)  # inst. GridSearchCV
# holdout-reasoning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
gm_cv.fit(X_train, y_train)  # Fitting to training data
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))  # 1.0 (-> i.e. Lasso regression!)
print("Tuned ElasticNet R squared: {}".format(r2))


#endregion (REGRESSION PIPELINES)





#region PIPELINES FOR MULTIPLE dtypes
# problem: pipeline steps for numeric and text preprocessing can't follow each other (output = input)
# solution: separate preprocessing pipelines, one main pipeline
# FunctionTransformer() and FeatureUnion()

from sklearn.preprocessing import FunctionTransformer
# funs to returns numeric/text columns from df
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x['numeric'], validate=False)
numeric_pipeline = Pipeline([  # numeric pl
    ('selector', get_numeric_data),
    ('imputer', Imputer())
])
text_pipeline = Pipeline([  # text pl
    ('selector', get_text_data),
    ('vectorizer', CountVectorizer())
])
pl = Pipeline([  # main pl
    ('union', FeatureUnion([
        ('numeric', numeric_pipeline),
        ('text', text_pipeline)])
    ),
    ('clf', OneVsRestClassifier(LogisticRegression()))
])
pl.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

# or
from sklearn.feature_extraction.text import HashingVectorizer
# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())])
                    ),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(
                        token_pattern=TOKENS_ALPHANUMERIC,
                        non_negative=True, norm=None,
                        binary=False, ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))])
                    )
                ])
            ),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))])
#endregion (PIPELINES FOR MULTIPLE dtypes)
