

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.metrics import roc_curve, roc_auc_score



#region regressors (illustrative)
X = df[[
    "x1", "x2", "x3", # cont. vars

    "F_sex",  # dummy vars (dummy trap!)
    # "M_sex",
    
    "<20_age", "20-29_age", "30-39_age",  # dummies
    # "40+_age",
    ]]
X = sm.add_constant(X)
#endregion



#region DISCRETE CHOICE MODELS

# pearson's chi2 independency test
tab = pd.crosstab(user5["x_disc"], user5['y_disc'])
# tab = tab.loc[:, ["None", "Some", "Marked"]]
table = sm.stats.Table(tab)
print(table.table_orig)  # original frequencies
print(table.fittedvalues)  # H0: perfectly independent
print(table.resid_pearson)  # pearson's residuals
chi2 = table.test_nominal_association()  # chi2
print(chi2)
print(table.chi2_contribs)  # individual contributions to chi2

# binary choice models
y = df.y_binary  # i.e. dummy DV
m_logit = sm.Logit(y, X).fit()  # option: Probit
print(m_logit.summary2())  # estimation summary
y_pred = m_logit.predict(X)  # fitted/predicted values
print(confusion_matrix(y, (y_pred > .5).astype(int)))

# nominal data models (not tested)
y = df.y_nominal  # DV
mn_logit = sm.MNLogit(y, X).fit()
print(mn_logit.summary2())  # estimation summary
y_pred = mn_logit.predict(X)  # fitted/predicted values
print(confusion_matrix(y, (y_pred > .5).astype(int)))

# count data models (w/ exposure!)
y = df.y_count  # DV

m_poiss = sm.Poisson(
    y, X, exposure=df['x_timespan'].values).fit()
print(m_poiss.summary2())

m_NB2 = sm.NegativeBinomial(
    y, X, loglike_method='nb2', exposure=df['x_timespan'].values).fit()
print(m_NB2.summary2())

m_NB1 = sm.NegativeBinomial(
    y, X, loglike_method='nb1', exposure=df['x_timespan'].values).fit()
print(m_NB1.summary2())

m_NBP = sm.NegativeBinomialP(
    y, X, exposure=df['x_timespan'].values).fit()
print(m_NBP.summary2())

#endregion



#region REGRESSION MODELS

# OLS family
y = df.y_count

m_OLS = sm.OLS(y, X).fit()  # i.i.d. errors
print(m_OLS.summary2())

m_GLS = sm.GLS(y, X).fit()  # arbitrary covariance between errors
print(m_GLS.summary2())

m_GLSAR = sm.GLSAR(y, X).fit()  # feasible GLS with autocorrelated AR(phi) errors
print(m_GLSAR.summary2())

# GMM family (not tested yet)
m_GMM = sm.GMM(y, X).fit()  # i.i.d. errors
print(m_GMM.summary2())

# ML family


# Instrumental Variables
m_IV2SLS = sm.IV2SLS(y, X, instrument=X_augmented).fit()
print(m_IV2SLS.summary2())

#endregion



#region TIME SERIES MODELS

# ARMA

# GARCH

# stationarity

#endregion
