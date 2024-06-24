import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import math
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Open and pre-process the dataset downloaded from https://www.kaggle.com/datasets/ander289386/cars-germany

df = pd.read_csv('autoscout24.csv')
df = df.drop_duplicates(keep='first')
df = df[(df['mileage'] > 0) & (df['mileage'] < 300000)]
df = df[(df['price'] > 100) & (df['price'] < 50000)]
df = df[(df['hp'] > 1) & (df['hp'] < 2000)]
df = df.dropna().reset_index(drop=True)

# Generate the sidebar to filter the data in the dashboard

with st.sidebar:
  
    container = st.container()
    container.write('**Art**')
    all = st.checkbox('Select all', key='all1', value=True)
    if all:
        selected_options_type = container.multiselect('**offerType:**',
        ['Demonstration', "Employee's car", 'New', 'Pre-registered', 'Used'],
        ['Demonstration', "Employee's car", 'New', 'Pre-registered', 'Used'],
        label_visibility='collapsed')
    else:
        selected_options_type = container.multiselect('**offerType:**',
        ['Demonstration', "Employee's car", 'New', 'Pre-registered', 'Used'],
        label_visibility='collapsed')

    container2 = st.container()
    container2.write('**Fuel**')
    all2 = st.checkbox('Select all', key='all2', value=True)
    if all2:
        selected_options_fuel = container2.multiselect('**Fuel:**',
        ['-/- (Fuel)', 'CNG', 'Diesel', 'Electric', 'Electric/Diesel', 'Electric/Gasoline', 'Ethanol', 'Gasoline', 'Hydrogen', 'LPG', 'Others'],
        ['-/- (Fuel)', 'CNG', 'Diesel', 'Electric', 'Electric/Diesel', 'Electric/Gasoline', 'Ethanol', 'Gasoline', 'Hydrogen', 'LPG', 'Others'],
        label_visibility='collapsed')
    else:
        selected_options_fuel = container2.multiselect('**Fuel:**',
        ['-/- (Fuel)', 'CNG', 'Diesel', 'Electric', 'Electric/Diesel', 'Electric/Gasoline', 'Ethanol', 'Gasoline', 'Hydrogen', 'LPG', 'Others'],
        label_visibility='collapsed')

    st.write('**Mileage**')
    mil_window = st.slider('Mileage', 0, 300000, 100000, label_visibility='collapsed')
    mil_max = mil_window

    st.write('**Year**')
    time_window = st.slider('Year', 2011, 2021, (2015, 2019), 1, label_visibility='collapsed')
    year_min = time_window[0]
    year_max = time_window[1]

# This code is to generate the dashboard

st.title('AutoScout')

st.header('Cars sold')
df = df[df['fuel'].isin(selected_options_fuel)]
df = df[df['offerType'].isin(selected_options_type)]
df = df[df['mileage'] <= mil_max]
df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]

makes = df['make'].unique()

top_make = df.groupby(['make'])['make'].count()
top_price = df.groupby(['make'])['price'].mean()
top_make_s = top_make.sort_values(ascending=False)
top_make_sorted = top_make_s[0:5] # Show top5 makes
top1 = pd.DataFrame({'Count': top_make_sorted})
top2 = pd.DataFrame({'Avg. price / €': top_price})
top = top1.join(top2)
top.index.names = ['Make']

col1, col2 = st.columns(2)
with col1:
    st.metric('Total', df['year'].count(), label_visibility='visible')
with col2:
    st.metric('No. makes', len(df['make'].unique()), label_visibility='visible')

df_fil = df.groupby(['year']).count()
fig1 = plt.figure(figsize=(6,3))
sns.lineplot(x=range(year_min, year_max+1), y=df_fil['make'], color="red")
plt.xticks(range(year_min, year_max+1))
plt.ylim([0,4500])
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.ylabel('No. cars')

col3, col4 = st.columns([2,1])
with col3:
    st.pyplot(fig1)
with col4:
    st.write(top)

fig2 = plt.figure(figsize=(4,4))
sns.regplot(data=df, x='mileage', y='price', line_kws=dict(color="r"), color=".3")
plt.xticks(rotation=45)
plt.xlabel('Mileage / km')
plt.ylabel('Price / €')

fig3 = plt.figure(figsize=(4,4))
sns.boxplot(data=df, x='year', y='mileage')
sns.pointplot(data=df, x='year', y='mileage', color="red")
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.ylabel('Mileage / km')

fig4 = plt.figure(figsize=(4,4))
sns.boxplot(data=df, x='year', y='price')
sns.pointplot(data=df, x='year', y='price', color="red")
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.ylabel('Price / €')

col5, col6, col7 = st.columns(3)
with col5:
    st.pyplot(fig2)
with col6:
    st.pyplot(fig4)
with col7:
    st.pyplot(fig3)

# This code is to predict a car's price of the top5 makes by the random forest algorithm

st.header('Predict price')

car_make = st.selectbox('Make', ('Volkswagen', 'Opel', 'Skoda', 'Renault', 'Ford'), index=0, placeholder='Make...', label_visibility='collapsed')

if car_make == 'Volkswagen':
    model_select = tuple(df[df['make']=='Volkswagen']['model'].unique())
    fuel_select = tuple(df[df['make']=='Volkswagen']['fuel'].unique())
    gear_select = tuple(df[df['make'] == 'Volkswagen']['gear'].unique())
    offer_select = tuple(df[df['make'] == 'Volkswagen']['offerType'].unique())
elif car_make == 'Opel':
    model_select = tuple(df[df['make']=='Opel']['model'].unique())
    fuel_select = tuple(df[df['make'] == 'Opel']['fuel'].unique())
    gear_select = tuple(df[df['make'] == 'Opel']['gear'].unique())
    offer_select = tuple(df[df['make'] == 'Opel']['offerType'].unique())
elif car_make == "Skoda":
    model_select = tuple(df[df['make']=='Skoda']['model'].unique())
    fuel_select = tuple(df[df['make'] == 'Skoda']['fuel'].unique())
    gear_select = tuple(df[df['make'] == 'Skoda']['gear'].unique())
    offer_select = tuple(df[df['make'] == 'Skoda']['offerType'].unique())
elif car_make == "Renault":
    model_select = tuple(df[df['make']=='Renault']['model'].unique())
    fuel_select = tuple(df[df['make'] == 'Renault']['fuel'].unique())
    gear_select = tuple(df[df['make'] == 'Renault']['gear'].unique())
    offer_select = tuple(df[df['make'] == 'Renault']['offerType'].unique())
elif car_mamek == "Ford":
    model_select = tuple(df[df['make']=='Ford']['model'].unique())
    fuel_select = tuple(df[df['make'] == 'Ford']['fuel'].unique())
    gear_select = tuple(df[df['make'] == 'Ford']['gear'].unique())
    offer_select = tuple(df[df['make'] == 'Ford']['offerType'].unique())

car_model = st.selectbox('Modell', model_select, index=0, placeholder='Model...', label_visibility='collapsed')
car_fuel = st.selectbox('Fuel', fuel_select, index=0, placeholder='Fuel...', label_visibility='collapsed')
car_gear = st.selectbox('Gear', gear_select, index=0, placeholder='Gear...', label_visibility='collapsed')
car_offer = st.selectbox('offerType', offer_select, index=0, placeholder='offerType...', label_visibility='collapsed')
car_mileage = st.text_input('Mileage', '50000', label_visibility='collapsed')
car_year = st.text_input('Year', '2015', label_visibility='collapsed')
car_hp = st.text_input('HP', '100', label_visibility='collapsed')
button = st.button('Run')

if button:
    realData = pd.DataFrame.from_records([{'mileage': car_mileage, 'make': car_make, 'model': car_model, 'fuel': car_fuel, 'gear': car_gear, 'offerType': car_offer, 'hp': car_hp, 'year': car_year}])

    makes_rf = ['Volkswagen', 'Opel', 'Ford', 'Skoda', 'Renault']
    df2 = pd.read_csv('autoscout24.csv')
    df2 = df2[df2['make'].isin(makes_rf)]
    df2 = df2.drop_duplicates(keep='first')
    df2 = df2[stats.zscore(df2.price) < 3]
    df2 = df2[stats.zscore(df2.hp) < 3]
    df2 = df2[stats.zscore(df2.mileage) < 3]
    df2 = df2.dropna()
    
    makeDummies = pd.get_dummies(df2['make'])
    df2 = df2.join(makeDummies)
    df2 = df2.drop('make', axis=1)
    modelDummies = pd.get_dummies(df2['model'])
    df2 = df2.join(modelDummies)
    df2 = df2.drop('model', axis=1)
    offerTypeDummies = pd.get_dummies(df2['offerType'])
    df2 = df2.join(offerTypeDummies)
    df2 = df2.drop('offerType', axis=1)
    gearDummies = pd.get_dummies(df2['gear'])
    df2 = df2.join(gearDummies)
    df2 = df2.drop('gear', axis=1)
    fuelDummies = pd.get_dummies(df2['fuel'])
    df2 = df2.join(fuelDummies)
    df2 = df2.drop('fuel', axis=1)
    
    df2['price'] = df2['price'].apply(lambda x: math.log(x)) # No linear correlation between price and mileage

    X = df2.drop(columns=['price'], axis=1)
    Y = df2['price']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, y_train)

    makeDummiesr = pd.get_dummies(realData.make)
    realData = realData.join(makeDummiesr['make'])
    realData = realData.drop('make', axis=1)
    modelDummiesr = pd.get_dummies(realData['model'])
    realData = realData.join(modelDummiesr)
    realData = realData.drop('model', axis=1)
    offerTypeDummiesr = pd.get_dummies(realData['offerType'])
    realData = realData.join(offerTypeDummiesr)
    realData = realData.drop('offerType', axis=1)
    gearDummiesr = pd.get_dummies(realData['gear'])
    realData = realData.join(gearDummiesr)
    realData = realData.drop('gear', axis=1)
    fuelDummiesr = pd.get_dummies(realData['fuel'])
    realData = realData.join(fuelDummiesr)
    realData = realData.drop('fuel', axis=1)
    
    fitModel = pd.DataFrame(columns=X.columns)
    fitModel = fitModel._append(realData, ignore_index=True)
    fitModel = fitModel.fillna(0)

    preds = forest_reg.predict(fitModel)

    st.metric('Predicted price / €', int(math.exp(preds)), label_visibility='visible')
