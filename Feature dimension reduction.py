import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr

#Import extracted radiomics features
file_path = r'C:\...\feature.xlsx'
df = pd.read_excel(file_path)
data_feature = df.iloc[:,1:]
label = df.iloc[:,0]

#Variance
# Calculate the variance of each column feature
variances = data_feature.var()

# Sort the variance results in descending order
sorted_variances = variances.sort_values(ascending=False)

# Calculate the number of features to be deleted (last 25%)
num_features_to_drop = int(len(sorted_variances) * 0.25)

# Retrieve the index of features to be retained
features_to_keep = sorted_variances.index[:-num_features_to_drop]

# Extract columns from DataFrame based on index
filtered_df = df[features_to_keep]

#Pearson
data_feature_std = StandardScaler().fit_transform(filtered_df)
data_feature_std = pd.DataFrame(data_feature_std)
data_feature_std.columns = data_feature.columns
drop_index = []
for i in range(data_feature_std.shape[1]):
    x1 = data_feature_std.iloc[:,i]
    m = i+1
    for j in range(m,data_feature_std.shape[1],1):
        #print(j)
        x2 = data_feature_std.iloc[:,j]
        pearson = pearsonr(x1,x2)[0]

        if pearson == 1 or pearson == -1:
            drop_index.append(pd.DataFrame(data_feature_std.iloc[:,j]).columns)
        if (pearson >0.75 and pearson < 1) or (pearson <-0.75 and pearson > -1):
            data_p1 = x1[label == 0]
            data_n1 = x1[label == 1]
            p_nomal_p_value1 = stats.shapiro(data_p1)[1]
            n_nomal_p_value1 = stats.shapiro(data_n1)[1]
            std_p_value1 = stats.levene(data_p1,data_n1)[1]
            if p_nomal_p_value1 > 0.05 and n_nomal_p_value1 > 0.05:
                if std_p_value1 > 0.05:
                    t_pvalue1 = stats.ttest_ind(data_p1,data_n1)[1]
                else:
                    t_pvalue1 = stats.ttest_ind(data_p1,data_n1,equal_var=False)[1]
            else:
                t_pvalue1 = stats.mannwhitneyu(data_p1,data_n1,alternative='two-sided')[1]

            data_p2 = x2[label == 0]
            data_n2 = x2[label == 1]
            p_nomal_p_value2 = stats.shapiro(data_p2)[1]
            n_nomal_p_value2 = stats.shapiro(data_n2)[1]
            std_p_value2 = stats.levene(data_p2,data_n2)[1]
            if p_nomal_p_value2 > 0.05 and n_nomal_p_value2 > 0.05:
                if std_p_value2 > 0.05:
                    t_pvalue2 = stats.ttest_ind(data_p2,data_n2)[1]
                else:
                    t_pvalue2 = stats.ttest_ind(data_p2,data_n2,equal_var=False)[1]
            else:
                t_pvalue2 = stats.mannwhitneyu(data_p2,data_n2,alternative='two-sided')[1]
                
            if t_pvalue1 >= t_pvalue2:
                drop_index.append(pd.DataFrame(data_feature_std.iloc[:,j]).columns)
            else:
                drop_index.append(pd.DataFrame(data_feature_std.iloc[:,i]).columns)
drop_index1 = np.unique(drop_index)
result = filtered_df.drop(drop_index1, axis=1)
data_pearson = pd.concat([result,label],axis = 1)