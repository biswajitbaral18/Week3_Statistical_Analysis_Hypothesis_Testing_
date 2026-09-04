# Week 3: Statistical Analysis and Hypothesis Testing in Python
# Dataset: week3_statistical_analysis_dataset.csv
# The dataset is self-generated for educational/business-analysis purposes.

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("week3_statistical_analysis_dataset.csv")

# Descriptive statistics
print(df.groupby("Promotion")["Purchase_Amount_INR"].agg(["count", "mean", "std", "median"]))

# H0: Promotion does not change the average purchase amount.
# H1: Promotion increases the average purchase amount.
promo = df.loc[df["Promotion"] == "Promotion", "Purchase_Amount_INR"]
no_promo = df.loc[df["Promotion"] == "No Promotion", "Purchase_Amount_INR"]

t_stat, p_value = stats.ttest_ind(promo, no_promo, equal_var=False)
print("\nWelch independent t-test")
print("t-statistic:", t_stat)
print("p-value:", p_value)

# 95% CI for difference in means
mean_diff = promo.mean() - no_promo.mean()
se = np.sqrt(promo.var(ddof=1)/len(promo) + no_promo.var(ddof=1)/len(no_promo))
df_welch = (promo.var(ddof=1)/len(promo) + no_promo.var(ddof=1)/len(no_promo))**2 / (
    (promo.var(ddof=1)/len(promo))**2/(len(promo)-1) +
    (no_promo.var(ddof=1)/len(no_promo))**2/(len(no_promo)-1)
)
t_critical = stats.t.ppf(0.975, df_welch)
ci_low = mean_diff - t_critical * se
ci_high = mean_diff + t_critical * se
print("Mean difference:", mean_diff)
print("95% CI:", (ci_low, ci_high))

# One-way ANOVA: purchase amount across categories
groups = [df.loc[df["Category"] == c, "Purchase_Amount_INR"]
          for c in ["Electronics", "Clothing", "Groceries"]]
f_stat, p_anova = stats.f_oneway(*groups)
print("\nOne-way ANOVA")
print("F-statistic:", f_stat)
print("p-value:", p_anova)

# Chi-square test: promotion status vs category
table = pd.crosstab(df["Promotion"], df["Category"])
chi2, p_chi, dof, expected = stats.chi2_contingency(table)
print("\nChi-square test")
print(table)
print("Chi-square:", chi2)
print("p-value:", p_chi)

# Effect size: Cohen's d
pooled_sd = np.sqrt(((len(promo)-1)*promo.var(ddof=1) +
                     (len(no_promo)-1)*no_promo.var(ddof=1)) /
                    (len(promo)+len(no_promo)-2))
cohen_d = mean_diff / pooled_sd
print("\nCohen's d:", cohen_d)

# Conclusion
alpha = 0.05
if p_value < alpha:
    print("\nDecision: Reject H0. Evidence supports the hypothesis that promotions increase average purchase amount.")
else:
    print("\nDecision: Fail to reject H0. Evidence is insufficient to conclude that promotions increase average purchase amount.")
