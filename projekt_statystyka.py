import numpy as np
import pandas as pd
import scipy.stats as stats

#wczytanie danych z kaggle
#ja akurat wybrałem San Francisco Airport Runway Use
#Late Night Departure Preferences -> https://www.kaggle.com/datasets/thedevastator/san-francisco-airport-runway-use?resource=download
data = pd.read_csv('dane.csv')

#obliczenie miar statystyki opisowej
mean = data['loudness'].mean()
std = data['loudness'].std()
mode = stats.mode(data['loudness'])
median = np.median(data['loudness'])
q1 = np.percentile(data['loudness'], 25)
q3 = np.percentile(data['loudness'], 75)
iqr = q3 - q1

#obliczenie klasycznego i pozycyjnego xtypu
xtyp_classical = stats.zscore(data['loudness'])
xtyp_positional = stats.rankdata(data['loudness'])

#obliczenie klasycznego i pozycyjnego wsp. zmienności
variance_classical = np.var(data['loudness'])
variance_positional = np.var(xtyp_positional)

#obliczenie asymetrii A1, A2 oraz A3
A1 = stats.skew(data['loudness'])
A2 = stats.kurtosis(data['loudness'])
A3 = stats.skewtest(data['loudness'])

#obliczenie korelacji pearsona pomiędzy wszystkimi parami cech statystycznych
corr_year_pop = stats.pearsonr(data['year'], data['loudness'])

#estymacje przedziałowe (z parametrem α = 0.025, 0.05 oraz 0.1) dla wszystkich cech
mean_conf_int_025 = stats.norm.interval(0.025, loc=mean, scale=std)
mean_conf_int_05 = stats.norm.interval(0.05, loc=mean, scale=std)
mean_conf_int_10 = stats.norm.interval(0.1, loc=mean, scale=std)

std_conf_int_025 = stats.norm.interval(0.025, loc=std, scale=std)
std_conf_int_05 = stats.norm.interval(0.05, loc=std, scale=std)
std_conf_int_10 = stats.norm.interval(0.1, loc=std, scale=std)

corr_conf_int_025 = stats.norm.interval(0.025, loc=corr_year_pop[0], scale=std)
corr_conf_int_05 = stats.norm.interval(0.05, loc=corr_year_pop[0], scale=std)
corr_conf_int_10 = stats.norm.interval(0.1, loc=corr_year_pop[0], scale=std)

#testy statystyczne dla 1 zmiennej (z parametrem α = 0.025, 0.05 oraz 0.1) dla wszystkich cech
mean_test_025 = stats.ttest_1samp(data['loudness'], popmean=mean)
mean_test_05 = stats.ttest_1samp(data['loudness'], popmean=mean)
mean_test_10 = stats.ttest_1samp(data['loudness'], popmean=mean)

std_test_025 = stats.ttest_1samp(data['loudness'], popmean=std)
std_test_05 = stats.ttest_1samp(data['loudness'], popmean=std)
std_test_10 = stats.ttest_1samp(data['loudness'], popmean=std)

corr_test_025 = stats.ttest_1samp(data['loudness'], popmean=corr_year_pop[0])
corr_test_05 = stats.ttest_1samp(data['loudness'], popmean=corr_year_pop[0])
corr_test_10 = stats.ttest_1samp(data['loudness'], popmean=corr_year_pop[0])

#testy statystyczne dla par zmiennych (z parametrem α = 0.025, 0.05 oraz 0.1) dla:
mean_test_2v_025 = stats.ttest_ind(data['year'], data['loudness'])
mean_test_2v_05 = stats.ttest_ind(data['year'], data['loudness'])
mean_test_2v_10 = stats.ttest_ind(data['year'], data['loudness'])

std_test_2v_025 = stats.ttest_ind(data['year'], data['loudness'])
std_test_2v_05 = stats.ttest_ind(data['year'], data['loudness'])
std_test_2v_10 = stats.ttest_ind(data['year'], data['loudness'])

corr_test_2v_025 = stats.ttest_ind(data['year'], data['loudness'])
corr_test_2v_05 = stats.ttest_ind(data['year'], data['loudness'])
corr_test_2v_10 = stats.ttest_ind(data['year'], data['loudness'])

print("Średnia arytmetyczna: ", mean)
print("Odchylenie standardowe: ", std)
print("Dominanta: ", mode)
print("Mediana: ", median)
print("Kwartyl dolny: ", q1)
print("Kwartyl górny: ", q3)
print("Odchylenie ćwiartkowe: ", iqr)
print("Klasyczny xtyp: ", xtyp_classical)
print("Pozycyjny xtyp: ", xtyp_positional)
print("Klasyczny wsp. zmienności: ", variance_classical)
print("Pozycyjny wsp. zmienności: ", variance_positional)
print("Asymetria A1: ", A1)
print("Asymetria A2: ", A2)
print("Asymetria A3: ", A3)
print("Korelacja pearsona pomiędzy wszystkimi parami cech statystycznych: ", corr_year_pop)
print("Estymacja przedziałowa dla średniej (α = 0.025): ", mean_conf_int_025)
print("Estymacja przedziałowa dla średniej (α = 0.05): ", mean_conf_int_05)
print("Estymacja przedziałowa dla średniej (α = 0.1): ", mean_conf_int_10)
print("Estymacja przedziałowa dla odchylenia standardowego (α = 0.025): ", std_conf_int_025)
print("Estymacja przedziałowa dla odchylenia standardowego (α = 0.05): ", std_conf_int_05)
print("Estymacja przedziałowa dla odchylenia standardowego (α = 0.1): ", std_conf_int_10)
print("Estymacja przedziałowa dla korelacji (α = 0.025): ", corr_conf_int_025)
print("Estymacja przedziałowa dla korelacji (α = 0.05): ", corr_conf_int_05)
print("Estymacja przedziałowa dla korelacji (α = 0.1): ", corr_conf_int_10)
print("Test statystyczny dla średniej (α = 0.025): ", mean_test_025)
print("Test statystyczny dla średniej (α = 0.05): ", mean_test_05)
print("Test statystyczny dla średniej (α = 0.1): ", mean_test_10)
print("Test statystyczny dla odchylenia standardowego (α = 0.025): ", std_test_025)
print("Test statystyczny dla odchylenia standardowego (α = 0.05): ", std_test_05)
print("Test statystyczny dla odchylenia standardowego (α = 0.1): ", std_test_10)
print("Test statystyczny dla korelacji (α = 0.025): ", corr_test_025)
print("Test statystyczny dla korelacji (α = 0.05): ", corr_test_05)
print("Test statystyczny dla korelacji (α = 0.1): ", corr_test_10)
print("Test statystyczny dla par zmiennych (α = 0.025): ", mean_test_2v_025)
print("Test statystyczny dla par zmiennych (α = 0.05): ", mean_test_2v_05)
print("Test statystyczny dla par zmiennych (α = 0.1): ", mean_test_2v_10)
print("Test statystyczny dla par zmiennych (α = 0.025): ", std_test_2v_025)
print("Test statystyczny dla par zmiennych (α = 0.05): ", std_test_2v_05)
print("Test statystyczny dla par zmiennych (α = 0.1): ", std_test_2v_10)
print("Test statystyczny dla par zmiennych (α = 0.025): ", corr_test_2v_025)
print("Test statystyczny dla par zmiennych (α = 0.05): ", corr_test_2v_05)
print("Test statystyczny dla par zmiennych (α = 0.1): ", corr_test_2v_10)

#Artur Kruszko