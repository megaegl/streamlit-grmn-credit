import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('german_credit_data.csv')

df["Checking account"].fillna(method="bfill",inplace=True)
df["Saving accounts"].fillna(method="bfill",inplace=True)

df['Housing'] = df['Housing'].str.replace('free','0')
df['Housing'] = df['Housing'].str.replace('own','1')
df['Housing'] = df['Housing'].str.replace('rent','2')
df['Housing'] = pd.to_numeric(df['Housing'])

df['Saving accounts'] = df['Saving accounts'].str.replace('little','0')
df['Saving accounts'] = df['Saving accounts'].str.replace('moderate','1')
df['Saving accounts'] = df['Saving accounts'].str.replace('quite rich','2')
df['Saving accounts'] = df['Saving accounts'].str.replace('rich','3')
df['Saving accounts'] = pd.to_numeric(df['Saving accounts'])

df['Checking account'] = df['Checking account'].str.replace('little','0')
df['Checking account'] = df['Checking account'].str.replace('moderate','1')
df['Checking account'] = df['Checking account'].str.replace('quite rich','2')
df['Checking account'] = df['Checking account'].str.replace('rich','3')
df['Checking account'] = pd.to_numeric(df['Checking account'])

df['Purpose'] = df['Purpose'].str.replace('radio/TV','0')
df['Purpose'] = df['Purpose'].str.replace('education','1')
df['Purpose'] = df['Purpose'].str.replace('furniture/equipment','2')
df['Purpose'] = df['Purpose'].str.replace('car','3')
df['Purpose'] = df['Purpose'].str.replace('business','4')
df['Purpose'] = df['Purpose'].str.replace('domestic appliances','5')
df['Purpose'] = df['Purpose'].str.replace('repairs','6')
df['Purpose'] = df['Purpose'].str.replace('vacation/others','7')
df['Purpose'] = pd.to_numeric(df['Purpose'])

df['Sex'] = df['Sex'].str.replace('male','1')
df['Sex'] = df['Sex'].str.replace('female','0')
df['Sex'] = df['Sex'].str.replace('fe1','2')
df['Sex'] = pd.to_numeric(df['Sex'])

df.rename(index=str, columns={'Saving accounts' : 'saving','Checking account' : 'checking','Credit amount' : 'credit'}, inplace=True)

x = df.drop(['Sex','Unnamed: 0'], axis=1)

#show data
st.header("Isi Dataset")
if st.button('Tampilkan dataset'):
    st.write(x)

#elbow
clusters=[]
for i in range(1,11):
  kmeans = KMeans(n_clusters=i).fit(x)
  clusters.append(kmeans.inertia_)

fig,ax=plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_xlabel('inertia')

st.header("Elbow Point")
st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot=st.pyplot()

st.sidebar.header("Clustering Nasabah Credit")
st.sidebar.subheader("Nama : Mega Dwi Rengganis")
st.sidebar.subheader("NIM : 211351081")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)
selectbox_x = st.sidebar.selectbox(
    "Input kolom 1 untuk visualisasi clustering",
    ('Age', 'Job', 'Housing', 'saving', 'checking', 'credit', 'Duration', 'Purpose')
)
if 'Age' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Age']
elif 'Job' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Job']
elif 'Housing' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Housing']
elif 'saving' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['saving']
elif 'checking' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['checking']
elif 'credit' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['credit']
elif 'Duration' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Duration']
elif 'Purpose' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Purpose']

selectbox_y = st.sidebar.selectbox(
    "Input kolom 2 untuk visualisasi clustering",
    ('Age', 'Job', 'Housing', 'saving', 'checking', 'credit', 'Duration', 'Purpose')
)
if 'Age' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Age']
elif 'Job' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Job']
elif 'Housing' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Housing']
elif 'saving' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['saving']
elif 'checking' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['checking']
elif 'credit' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['credit']
elif 'Duration' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Duration']
elif 'Purpose' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Purpose']

def k_means(n_clust,x_plot,y_plot):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]

    pl = sns.scatterplot(x=x_plot, y=y_plot,hue=x["Labels"], palette= pal)
    pl.set_title(f"Clustering berdasarkan {selectbox_x} dan {selectbox_y}")
    for label in x['Labels']:
        pl.annotate(label,
            (x[x['Labels']==label]['credit'].mean(),
            x[x['Labels']==label]['Duration'].mean()),
            horizontalalignment = 'center',
            verticalalignment = 'center',
            size = 20, weight='bold',
            color='black')
    st.header("Cluster Plot")
    st.pyplot()
    st.write(x)
    plt.legend()
    plt.show()

k_means(clust,x_plot,y_plot)