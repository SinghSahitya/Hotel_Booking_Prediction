import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from calendar import month_name
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings

filterwarnings('ignore')

df = pd.read_csv('Data/hotel_bookings.csv')

def sort_month(df, colname):
    month_dict = {j:i for i, j in enumerate(month_name)}
    df['month_num'] = df[colname].apply(lambda x: month_dict[x])
    return df.sort_values(by='month_num').reset_index().drop(['index', 'month_num'], axis=1)


def data_clean(df):
    df.fillna(0, inplace=True) #zero impuatation
    # print(df.isna().sum())

data_clean(df)

# list = ['children', 'adults', 'babies']

# for i in list:
#     print(f"{i} has unique values {df[i].unique()}")

filtered_data = (df['children'] == 0) & (df['adults']==0) & (df['babies']==0)
final_data = df[~filtered_data]

country_wise_data = final_data[final_data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns = ['Country', 'No. of Guests']
# print(country_wise_data)

map = px.choropleth(country_wise_data , locations=country_wise_data['Country']
                                    ,color=country_wise_data['No. of Guests']
                                    ,hover_name=country_wise_data['Country']
                                    ,title='Guest Countries')

map.show()

data = final_data[final_data['is_canceled'] == 0]

plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',
            y="adr",
            hue='hotel',data=data)

plt.title("Price of room per nigth per person",fontsize=16)
plt.xlabel("Room Type")
plt.ylabel("Price [EUR]")
plt.legend(loc="upper right")
plt.ylim(0,600)
plt.show()

data_resort = final_data[(final_data['hotel']=='Resort Hotel')&final_data['is_canceled']==0]
data_city = final_data[(final_data['hotel']=='City Hotel')&final_data['is_canceled']==0]

resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()

final = resort_hotel.merge(city_hotel, on='arrival_date_month')
final.columns = ['month', 'price_for_resort_hotel','price_for_city_hotel']
sort_month(final, "month")

final.plot(kind='line',x='month', y = ['price_for_resort_hotel', 'price_for_city_hotel'])

rush_resort = data_resort["arrival_date_month"].value_counts().reset_index()
rush_resort.columns = ['month', 'no. of guests']

rush_city = data_city["arrival_date_month"].value_counts().reset_index()
rush_city.columns = ['month', 'no. of guests']

final_rush = rush_resort.merge(rush_city, on='month')
final_rush.columns = ['month', 'people_in_resort_hotel', 'people_in_city_hotel']
sort_month(final_rush, "month")

final_rush.plot(kind='line', x='month', y=['people_in_resort_hotel', 'people_in_city_hotel'])
data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay = stay.iloc[0:3]
stay = stay.rename(columns={'is_canceled':'Number_of_stays'})

sns.barplot(x="total_nights", y="Number_of_stays", hue='hotel', hue_order=["City Hotel", "Resort Hotel"], data=stay)

#feature selection

correlation = final_data.corr()
correlation = correlation['is_canceled'][1:]
# print(correlation.abs().sort_values(ascending=False))


list_not = ['days_in_waiting_list', 'arrival_date_year']

num_features = [col for col in final_data.columns if final_data[col].dtype != 'O' and col not in list_not]
data_num = final_data[num_features]
# print(num_features)

cat_not = ["country", "reservation_status", "booking_ch", "assigned_room_type", "days_in_waiting_list"]
cat_features = [col for col in final_data.columns if final_data[col].dtype == 'O' and col not in cat_not]

data_cat = final_data[cat_features]
data_cat['reservation_status_date'] = pd.to_datetime(data_cat['reservation_status_date'])
data_cat['year'] = data_cat['reservation_status_date'].dt.year
data_cat['month'] = data_cat['reservation_status_date'].dt.month
data_cat['day'] = data_cat['reservation_status_date'].dt.day
data_cat.drop('reservation_status_date', axis=1, inplace=True)

#mean_encoding
def mean_encode(df, col, mean_col):
    df_dict = df.groupby([col])[mean_col].mean().to_dict()
    df[col] = df[col].map(df_dict)
    return df

data_cat['cancellation'] = final_data['is_canceled']

for col in data_cat.columns[0:8]:
    data_cat = mean_encode(data_cat, col, 'cancellation')


data_cat.drop('cancellation', axis=1, inplace=True)

#preparing_data
dataframe = pd.concat([data_num, data_cat], axis=1)

#handling_outliers
def handle_outlier(col):
    dataframe[col] = np.log1p(dataframe[col])

handle_outlier('lead_time')
handle_outlier('adr')
dataframe.dropna(inplace=True)

#feature_importance
Y = dataframe['is_canceled']
X = dataframe.drop('is_canceled', axis=1)

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X, Y)
cols = X.columns
selected_feature = cols[(feature_sel_model.get_support())]
# print(selected_feature)

#building_models
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state=45)

models = []

models.append(("Logistic Regression", LogisticRegression()))
models.append(("Naive Bayes", GaussianNB()))
models.append(("Random Forest", RandomForestClassifier()))
models.append(("Decision Tree", DecisionTreeClassifier()))


for name, model in models:
    print(name)
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    print(confusion_matrix(model_pred, Y_test))
    print(accuracy_score(model_pred, Y_test))
    print("\n")