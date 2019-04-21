

# Imports
import matplotlib.pyplot as plt
import seaborn as sns








#region 1 DISCRETE VAR

# .value_counts(): balanced classes?
df["x_disc"].value_counts()

# lineplot count variable x
_ = plt.plot(
    df["x_disc"].value_counts().sort_index().index,
    df["x_disc"].value_counts().sort_index())
_ = plt.title("Title")
_ = plt.xlabel("xlabel")
_ = plt.ylabel("ylabel")
plt.show()

# lineplots over time of several discrete variables in same graph, same scale
_ = plt.style.use("bmh")  # "ggplot2", "fivethirtyeight", "seaborn", etc.
_ = sns.lineplot(
    x="x_time", y="x1_disc", 
    data=df, label="x1_label", color="royalblue")
_ = sns.lineplot(
    x="x_time", y="x2_disc",  
    data=df, label="x2_label", color="crimson")
_ = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize="small")
_ = plt.xticks(rotation=60)
_ = plt.title("Title\n")
_ = plt.ylabel("ylabel", fontsize="small")
_ = plt.xlabel("xlabel", fontsize="small")
plt.show()

#endregion






#region 1 CONTINUOUS VAR

# histogram
_ = sns.distplot(df["x_cont"], bins=None, hist=True, kde=False, rug=False)
plt.show()

# kernel density estimation
_ = sns.kdeplot(df["x_cont"], shade=True, legend=True)
plt.show()

# kde of x_cont with hue x_disc
grid = sns.FacetGrid(df, hue="x_disc", col_order=range(-4,0)[::-1])
_ = grid.map(sns.kdeplot, "x_cont")
_ = grid.add_legend()
plt.show()

# pdf and cdf on two different plots
fig, axes = plt.subplots(nrows=2, ncols=1)  # !!!
_ = df["x_cont"].plot(ax=axes[0], kind="hist", bins=30, density=True, range=(0,10))  # Plot the PDF
_ = plt.show()
_ = df.["x_cont"].plot(ax=axes[1], kind="hist", bins=30, density=True, cumulative=True, range=(0,10))  # Plot the CDF
plt.show()

#endregion






#region 2 DISCRETE VARS

# contingency tables! test of independence
tab = pd.crosstab(df["x1_disc"], df['x2_disc'])
table = sm.stats.Table(tab)
print(table.table_orig)  # original freq.
print(table.fittedvalues)  # freq. if independent
print(table.resid_pearson)  # pearson residuals
chi2 = table.test_nominal_association()
print(chi2)  # chi2
print(table.chi2_contribs)  # chi2 contributions

# pivot tables (values of 2 vars on 2 columns) count, frequencies
df.pivot_table(
    index="x1_disc",
    columns="x2_disc",
    values="x3",
    aggfunc="count")

# groupby (one observation per row)

# endregion







#region 1 DISCRETE x 1 CONTINUOUS VAR

# boxplot x_disc x x_cont (hue)
_ = sns.boxplot(x="x1_disc", y="x_cont", hue="x2_disc", data=df)
plt.show()

# swarmplot!
_ = sns.swarmplot(x="x1_disc", y="x_cont", data=df, hue="x2_disc")
plt.show()

# boxplot w/ swarmplot combined
_ = sns.boxplot(x="x_disc", y="x_cont", data=df)
_ = sns.swarmplot(x="x_disc", y="x_cont", data=df, color=".25")
plt.show()

# stripplots!
_ = plt.subplot(2,1,1)  # 2 vertically aligned plots, 1st plot
_ = sns.stripplot(
    x="x_disc",
    y="x_cont",
    data=df)
_ = plt.subplot(2,1,2)  # 2 vertically aligned plots, 2nd plot
_ = sns.stripplot(
    x="x_disc",
    y="x_cont",
    data=df,
    jitter=True, size=3)  # jitter, smaller points
plt.show()

# violinplot!
_ = plt.subplot(2,1,1)
_ = sns.violinplot(x="x_disc", y="x_cont", data=df)
_ = plt.subplot(2,1,2)  
_ = sns.violinplot(x="x_disc", y="x_cont", data=df, color="lightgray", inner=None)  # same violin plot w/ diff color w/o inner annotations
_ = sns.stripplot(x="x_disc", y="x_cont", data=df, jitter=True, size=1.5)  # Overlay strip plot on the violin plot
plt.show()

#endregion











#region 2 CONTINUOUS VARS

# scatterplot
_ = plt.scatter(df['x1_cont'], df["x2_cont"], label='label', color='red', marker='o')
_ = plt.legend(loc="upper right")
_ = plt.xlabel('xlabel')
_ = plt.ylabel('ylabel')
_ = plt.title('Title')
plt.show()

# scatterplot with linear regression
_ = sns.lmplot(x='x1_cont', y='x2_cont', data=df)
plt.show()
# residuals of regression
_ = sns.residplot(x='x1_cont', y='x2_cont', data=df, color='green')
plt.show()

# scatterplot with linear and nonlinear regressions
_ = plt.scatter(df['x1_cont'], df["x2_cont"], label='label', color='red', marker='o')
_ = sns.regplot(x='x1_cont', y='x2_cont', data=df, color="blue", scatter=None, label="order 1", order=1)  # plot in blue a linear regression of order 1 
_ = sns.regplot(x="x1_cont", y="x2_cont", data=df, color="green", scatter=None, label="order 2", order=2)  # plot in green a linear regression of order 2
_ = plt.legend(loc="upper right")
_ = plt.xlabel('xlabel')
_ = plt.ylabel('ylabel')
_ = plt.title('Title')
plt.show()

# 2-d histogram
_ = plt.hist2d(df["x1_cont"], df["x2_cont"], bins=(20, 20), range=((0, 10), (0, 1)))
_ = plt.colorbar()  # colorbar
_ = plt.xlabel('xlabel')
_ = plt.ylabel('ylabel')
_ = plt.title('Title')
plt.show()

# 2-d histogram with hex. bins
_ = plt.hexbin(df["x1_cont"], df["x2_cont"], gridsize=(20, 10), extent=(0, 10, 0, 1))
_ = plt.colorbar()
_ = plt.xlabel('xlabel')
_ = plt.ylabel('ylabel')
_ = plt.title('title')
plt.show()

# scatter w/ hist for each variable
_ = sns.jointplot(x='x1_cont', y='x2_cont', data=df)
plt.show()

# scatter (kde) w/ hist for each var
_ = sns.jointplot(x='x1_cont', y='x2_cont', data=df, kind="kde")  # other options: "kde", "reg", "resid", "scatter", "hex"
plt.show()

# pairwise scatters plus histograms!
_ = sns.pairplot(df, vars=["x1_cont", "x2_cont", "x3_cont"], kind="scatter")
plt.show()

# pairwise regressions plus kde hue x_disc!!!
_ = sns.pairplot(data=df, vars=["x1_cont", "x2_cont", "x3_cont"], kind="reg", hue="x_disc")
plt.show()


#endregion
