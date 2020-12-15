import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
def main():
    activities=['About','Toxic Comment Classification System','Developer']
    option=st.sidebar.selectbox('Menu Bar:',activities)
    
       
    
    st.title("Toxic Comment Classification System")
    f=open("model.pkl", "rb")
    model = pickle.load(f)
    v= pickle.load(f)
    st.header('Input Comment')
    text = st.text_input("Input text")
    lis=[]
    for i in range(6):
        lis.append(model[i].predict_proba(v.transform([text]))[:, 1])
    list=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    st.header('Output')
    for i in range(6):
        st.subheader(list[i])
        st.write(str(lis[i]))
    st.write('---')
    
    
          
if __name__ == '__main__':
    main()
