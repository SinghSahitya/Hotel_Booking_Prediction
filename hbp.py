import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('Data/hotel_bookings.csv')


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

# map.show()

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
# plt.show()

data_resort = final_data[(final_data['hotel']=='Resort Hotel')&final_data['is_canceled']==0]
data_city = final_data[(final_data['hotel']=='City Hotel')&final_data['is_canceled']==0]

resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel = data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()

final = resort_hotel.merge(city_hotel, on='arrival_date_month')
final.columns = ['month', 'price_for_resort_hotel','price_for_city_hotel']
# print(final)