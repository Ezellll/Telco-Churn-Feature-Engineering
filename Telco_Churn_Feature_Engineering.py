import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###########################################################
# Görev 1 : EDA
###########################################################

# Adım 1:

def load():
    data = pd.read_csv("feature_engineering/Telco-Customer-Churn/Telco-Customer-Churn.csv")
    return data

df = load()
df.columns
df.shape
df.head()
df.info()
df.describe().T

# TotalCharges değişkenini string formatına çevirip içerisinde bulunan boşlukları sildildikten sonra
# nümeric değişken olarak tanımladım.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip())


# Adım 2: Numerik ve kategorik değişkenleri yakalanmıştır.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # Nmerik görünülü kategorikleri çıkarttık
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapılmıştır.

# Nümerik
df[num_cols].head()
df[num_cols].describe().T
# Kategorik
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

for col in cat_cols:
    cat_summary(df, col)

# Adım 4
# Hedef değişken analizi yapılmıştır.

df["_Churn"] = np.where(df["Churn"] == "Yes", 1, 0)
for col in cat_cols:
    print(df.groupby(col)["_Churn"].mean(), "\n")
df.drop("_Churn", inplace=True, axis=1)

df.groupby("Churn")[num_cols].mean()
df.head()

# Adım 5: Aykırı gözlem analizi yapılmıştır.

df[num_cols].describe([.25, .5, .75, .90, .95, .99]).T

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


# Aykırı değer gözlenmemektedir.

# Adım 6: Eksik gözlem analizi yapılmıştır.

df.isnull().values.any()
df.isnull().sum()
df.notnull().sum()
df.shape

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

# Adım 7: Korelasyon analizi yapılmıştır.

df.corr()

###############################################################################
# Görev 2 :  Feature Engineering
###############################################################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemler yapılır.

# TotalCharges değişkeni içerisindeki boş değerlerin, tenure(Müşterinin şirkette kaldığı ay sayısı) incelendiğinde
# bu müşterilerin yeni müşteriler olduğu ve TotalCharges değişkenlerinin bu nedenle 0 olduğu anlaşılmaktadır.
# Veri setinden TotalChargesdeğişkenini silerek yanlış veya yanıltıcı sounçlar elde edilebiliriz bundan dolayı
# boş değişkenleri 0 sabit değeri ile doldurmak veri seti için doğru bir karar olacaktır.

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# Numeric değişkenlere bakıldığı zaman herhangi bir aykırı değer gözlenmemektedir.



# Adım 2: Yeni değişkenler oluşturulur.

df.loc[(df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes"), ["Streaming"]] = "Yes"
df.loc[~((df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")), ["Streaming"]] = "No"


df.loc[(df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes"), ["Online"]] = "Yes"
df.loc[~((df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")), ["Online"]] = "No"


# Adım 3: Encoding işlemlerini gerçekleştirilir.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapılır.

mms = MinMaxScaler()
df[num_cols] = mms.fit_transform(df[num_cols])

# Adım 5: Model oluşturulur.

y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
