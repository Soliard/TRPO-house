#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('House_data.csv')
df
sns.boxplot(y='price', x='bedrooms', data=df)


# Откинем id (не несет полезной информации), zipcode (код зоны, где расположен дом) и координаты (lat & long), так как корректное преобразование географических данных слишком специфично.

# In[2]:


df = df.drop(['id', 'zipcode', 'lat', 'long'], axis=1)
df


# Теперь посмотрим на дату объявления. Формат даты задан YYYYMMDDT000000, в целом ее тоже можно было бы удалить из датасета, но у нас есть поля год постройки (yr_built) и год последнего ремонта (yr_renovated), которые заданы в в формате года (YYYY), что не очень информативно. Оперируя датой объявления можно преобразовать год в возраст вычитанием (год объявления — год постройки / год ремонта). Отметим по части домов год ремонта стоит 0, и, предположив, что это означает отсутствие ремонта постройки, заменим нули в году ремонта на год постройки, предварительно убедившись, что в данных отсутствуют некорректные записи, где год ремонта меньше года постройки:

# In[3]:


df[(df['yr_renovated'] < df['yr_built']) & df['yr_renovated'] != 0]


# Таких некорректных записей нет

# In[4]:


df.loc[df['yr_renovated']==0, ['yr_renovated']]=df['yr_built']
df['yr_built']=df['date'].str[0:4].astype(int)-df['yr_built']
df['yr_renovated']=df['date'].str[0:4].astype(int)-df['yr_renovated']
df=df.drop('date', axis=1)
df


# Следующим параметром проанализируем цену и воспользуемся для этого «Ящиком с усами» (Box plot). Ящик с усами – простой и удобный график, показывающий одномерное распределение вероятностей, или, проще говоря, концентрацию данных. Отрисовывает медиану (линия в центре), верхний и нижний квартили (стороны ящика), края статистически значимой выборки («усы») и выбросы (точки за «усами»). Легко понять по картинке на нормальном распределении (справа). График позволяет быстро оценить где располагается большая часть данных (50% находятся внутри ящика), их симметричность (смещение медианы к одной из сторон ящика и/или длина «усов») и степень разброса – дисперсию (размеры ящика, размеры усов и количество точек-выбросов).

# In[5]:


sns.boxplot(y = df['price'])


# In[6]:


sns.boxplot(y='price', x='bedrooms', data=df)


# Из графика сразу видно наличие экстремальных значений price и bedrooms (только представьте дом с 33 спальнями! J). Наличие таких значений (иначе называемых как выбросы) в целевом признаке price часто приводит к переобучению модели, так именно они будут давать большую ошибку, которую алгоритмы стараются минимизировать. Из графика видно, что большая часть (если посчитать – 93,22%)  лежит в диапазоне 0-1млн, а свыше 2млн – всего 198 значений (0,92%). От 1% датасета можно избавиться практически безболезненно, поэтому вызвав простой просмотр 217 записей предварительно отсортировав по цене, увидим искомую отметку price в 1 965 000 и удалим все что выше этой цены.

# In[7]:


df.sort_values (by='price', ascending=False).head(217) 
df = df[df['price']<=1965000]


# In[8]:


df


# Подумаем немного над признаком bedrooms. Мы видим 13 домов с bedrooms = 0, а также странную запись о доме с 33 bedrooms. Поступим также как и с price, удалив нули из bedroms (а заодно и bathrooms):

# In[9]:


df = df[(df['bedrooms'] !=0 ) & (df['bathrooms'] != 0)]
df


# Касательно дома с 33 спальнями – учитывая цену, можно предположить что это опечатка и спален на самом деле 3. Сравним жилую площадь этого дома (1620) со средней жилой площадью домов с 3 спальнями (1798,2), что ж вероятно наша догадка верна, поэтому просто изменим это значение на 3 и еще раз построим предыдущий box plot:

# In[10]:


df.loc[df['bedrooms']==33,['bedrooms']]=3 
sns.boxplot(y='price', x='bedrooms', data=df)


# Чтож, значительно лучше. Аналогично bedrooms посмотрим и на bathrooms. Нулевые значения мы удалили, другие экстремальные значения в поле отсутствуют:

# In[11]:


sns.boxplot(y='bathrooms', x='bedrooms', data=df)


# В полях sqft_living, floors, waterfront, view, condition, grade, sqft_living15 также все значения более-менее реальны, их трогать не будем:

# In[12]:


plt.rcParams['figure.figsize']=2,3
sns.boxplot(y='sqft_living', data=df)
plt.show()
sns.boxplot(y='floors',color='#2ecc71', data=df)
plt.show()
sns.boxplot(y='sqft_living15',color='#9b59b6', data=df)
plt.show()
plt.rcParams['figure.figsize']=4,4
sns.boxplot(y='price', x='waterfront', data=df)
plt.show()
sns.boxplot(y='price', x='view' , data=df)
plt.show()
sns.boxplot(y='price', x='condition' , data=df)
plt.show()
sns.boxplot(y='price', x='grade' , data=df)
plt.show()


# А вот с sqft_lot и sqft_lot15 нужно что-то придумать и из-за больших значений вполне подойдет логарифмирование:

# In[13]:


df['sqft_lot']=np.log(df['sqft_lot'])
df['sqft_lot15']=np.log(df['sqft_lot15'])


# sqft_above и sqft_basement – составные части sqft_living, поэтому также трогать их не будем.
# На этом с предварительным анализом мы закончим и посмотрим на тепловую карту корреляций:

# In[14]:


plt.rcParams['figure.figsize']=13,12
sns.heatmap(df.corr(),  cmap = 'viridis',annot = True)


# Изучив карту корреляций видим, что иногда признаки сильно коррелированы между собой, поэтому удалим часть признаков с высокой корреляцией – sqft_lot15 (оставим sqft_lot),  yr_built (оставим yr_renovated), sqft_above (sqft_living)
# 
# На этом закончим работу с данными и перейдем к созданию модели ЛИНЕЙНОЙ РЕГРЕССИИ.
# Все необходимые нам модели содержаться в библиотеке sklearn.
# 
# Для начала отделим целевую переменную от остальных данных для обучения, а также разделим выборки на обучающую (70%) и тестовую (30%, на которой мы проверим как работает модель):

# In[15]:


Y=df['price']
X=df.drop ('price',axis=1) 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size = 0.3, shuffle = True)


# Также из sklearn для оценки модели загрузим 3 метрики — mean_absolute_error (средняя абсолютная ошибка), mean_squared_error (Среднеквадратическое отклонение), r2_score (коэффициент детерминации):

# In[16]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[17]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()     #Создаем модель
LR.fit(X_train, Y_train)    #Обучаем модель
Y_LR = LR.predict(X_test)    #Предсказываем значения для выборки    
print ('MAE:', round (mean_absolute_error(Y_test, Y_LR),3))    #Метрики
print ('√MSE:', round (mean_squared_error(Y_test, Y_LR)**(1/2),3))
print ('R2_score:', round (r2_score(Y_test, Y_LR),3))


# In[73]:


import pickle
pickle.dump(LR, open('model.pkl','wb')) 


# In[ ]:




