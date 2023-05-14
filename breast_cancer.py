import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
from openai_chat_gpt_code import *
from database import *
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report




# --------------------------------------Breast Cancer Prediction Coding Section Starts------------------------------------




cancer_df=pd.read_csv(r'breast cancer.csv')
# pd.set_option('display.max_columns',None)

print(cancer_df.tail(6))
print(cancer_df.describe())

target_name='target'
y=cancer_df[target_name]
X=cancer_df.drop(target_name,axis=1)
print(X.head())
print(y.head())

## Applying Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(X)

SSX=scaler.transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(SSX,y,test_size=0.2,random_state=7)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)


print("Train Accuracy Of Logistic Regression",lr.score(X_train,y_train)*100)
print("Test Accuracy of LRegression",lr.score(X_test,y_test)*100)
print("Accuracy Score of Logistic Regression",accuracy_score(y_test,lr_pred)*100)

## Saving the LR Model 
pickle.dump(lr,open('Logistic_Breast.pickle','wb'))
# ---------------------------------KNN------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)


print("Train Accuracy Of KNN",knn.score(X_train,y_train)*100)
print("Test Accuracy of KNN",knn.score(X_test,y_test)*100)
print("Accuracy Score of KNN",accuracy_score(y_test,knn_pred)*100)

# Saving the KNN model
pickle.dump(knn,open('KNN_Breast.pickle','wb'))

## Naive Bayes

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
nb_pred=nb.predict(X_test)


print("Train Accuracy Of Naive Baiyes",nb.score(X_train,y_train)*100)
print("Test Accuracy of Naive Bayes",nb.score(X_test,y_test)*100)
print("Accuracy Score of Naive Baiyes",accuracy_score(y_test,nb_pred)*100)

## Saving The Naive Baiyes model
pickle.dump(nb,open('NB_Breast.pickle','wb'))

## SVM

from sklearn.svm import SVC
sv=SVC(probability=True)
sv.fit(X_train,y_train)
sv_pred=sv.predict(X_test)



print("Train Accuracy Of SVM",sv.score(X_train,y_train)*100)
print("Test Accuracy of SVM",sv.score(X_test,y_test)*100)
print("Accuracy Score of SVM",accuracy_score(y_test,sv_pred)*100)

## Saving SVM Model 
pickle.dump(sv,open('SVM_Breast.pickle','wb'))

# --------------------------------Decision Tre with entropy--------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)



print("Train Accuracy Of Decision Tree",dt.score(X_train,y_train)*100)
print("Test Accuracy of Decision Tree",dt.score(X_test,y_test)*100)
print("Accuracy Score of Decision Tree",accuracy_score(y_test,dt_pred)*100)

## Saving the Decision Tree Model
pickle.dump(dt,open('Decision_Breast.pickle','wb'))


# ----------------------------------------Random Forest with entropy as criteria------------------------------------

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='entropy')
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)



print("Train Accuracy Of Random Forest",rf.score(X_train,y_train)*100)
print("Test Accuracy of Random Forest",rf.score(X_test,y_test)*100)
print("Accuracy Score of Random Forest",accuracy_score(y_test,rf_pred)*100)

## Saving Random Forest Model
pickle.dump(rf,open('Random_Breast.pickle','wb'))


# --------------------------------------Breast Cancer Prediction Coding Section Starts------------------------------------






# -------------------------------------------------Web page section starts---------------------------------------------------



def appointment():
    import pandas as pd
    import streamlit as st
    from datetime import datetime
    import warnings
    import disease_new
    from database import doctor_appointment,disease_add,disease_return,alterdiseasename,doctor_available_for_disease,verifyemail,mailverify,add_appointment1,add_appointment2,add_appointment3,add_appointment4,book_app
    warnings.filterwarnings("ignore")
    from otpcode import emailappointment

    global Doctor_Name
    global Disease_Name
    global disease

    appointment_section_title='<p style="font-family:Georgia; color:##00FFFF; font-size: 40px;text-align:center;">Available Doctors For Your Disease.</p>'
    st.markdown(appointment_section_title,unsafe_allow_html=True)
    Disease_Name=disease_return()
    st.success(Disease_Name)
    b=doctor_appointment(Disease_Name)
    if b==0:
        st.error('No Doctor Available For Your Disease At The Moment.')
    else:
        a=pd.DataFrame(b,columns=['Name','Email','Contact','Specialization_In_Disease','City'])
        st.dataframe(a,height=100,width=5000)

        appointment_book_title='<p style="font-family:Georgia; color:##00FFFF; font-size: 29px;text-align:center;">Book Your Appointment.</p>'
        st.markdown(appointment_book_title,unsafe_allow_html=True)

        if st.checkbox("Book"):
            
            doctor=doctor_available_for_disease(Disease_Name)
            
            # st.success()
            # st.success(doctor)
            Doctor=[]
            for i in doctor:
                Doctor.append(i[0])
                # st.success(i[0])
            Doctor.insert(0,"None")
            Doctor_Name=st.selectbox('Available Doctor',Doctor)
            mail=st.text_input('Enter Your Registered Email-Id')
            if mail=='':
                pass
            else:
                phonecome=verifyemail(mail)
                
                if phonecome==0:
                    st.warning("No Email found ! Please Register Yourselves and come again.")
                else:
                    emailverify=mailverify(phonecome)
                    if mail==emailverify:
                        st.success("Success")
                        if Doctor_Name=='None':
                            pass
                        else:
                            p1=add_appointment1(mail)  #Name
                            p2=add_appointment2(mail)   #Number
                            d1=add_appointment3(Doctor_Name) # DEmail   
                            d2=add_appointment4(Doctor_Name) # DNumber
                            # d3=add_appointment5(Disease_Name) # Disease
                            # t=add_appointment5() # Disease
                            t=datetime.now()
                            book_app(p1,mail,p2,Doctor_Name,d1,d2,Disease_Name,t)
                            st.success('Appointment Booked Successfully.')
                            st.success("Visit Profile Section for your booking details")

                            emailappointment(mail,Doctor_Name)
                            st.success("Visit Your Registered Email-Id  for your appointment details")
                            st.success("Thank you for choosing us, Doctor will reach you soon via phonecall or videocall")
                            alterdiseasename()
                            # st.success(c)
                    else:
                        st.warning("Wrong Email")
    
b='<h1 class="heading" style="text-align: center;font-size:3rem;color:#333;padding:0.1rem;margin:2rem 0;background:pink;"><span style="color:#F55887;">Breast Cancer </span>Detection</h1>'
st.markdown(b,unsafe_allow_html=True)



selected = option_menu(None, ["Plot", "Predict",'DOCTOR APPOINTMENT'], 
      icons=['bar-chart', "activity", 'person-lines-fill'], 
      menu_icon="cast", default_index=0, orientation="horizontal")
selected  

if selected=='Predict':
    upper_header_title = '<p style="font-family:Georgia; color:##00FFFF; font-size: 29px;text-align:center;">Whether suffering from Breast Cancer? Predict IT..</p>'
    st.markdown(upper_header_title, unsafe_allow_html=True)
    radius=st.text_input("Mean Radius")
    radius=str(radius)

    texture=st.text_input("Mean Texture")
    texture=str(texture)

    perimeter=st.text_input("Mean Perimeter")
    perimeter=str(perimeter)

    area=st.text_input("Mean Area")
    area=str(area)

    smootness=st.text_input("Mean Smootness")
    smootness=str(smootness)
    compactness=st.text_input("Mean Compactness")
    compactness=str(compactness)

    concavity=st.text_input("Mean Concave Points")
    concavity=str(concavity)

    concave_points=st.text_input("mean concave points")
    concave_points=str(concave_points)

    symmetry=st.text_input("Mean Symmetry")
    symmetry=str(symmetry)

    fractal_dimension=st.text_input("Mean Fractal Dimension")
    fractal_dimension=str(fractal_dimension)

    radius_error=st.text_input("Radius Error")
    radius_error=str(radius_error)

    texture_error=st.text_input("Texture error")
    texture_error=str(texture_error)

    perimeter_error=st.text_input("Perimeter error")
    perimeter_error=str(perimeter_error)

    area_error=st.text_input("area error")
    area_error=str(area_error)

    smoothness_error=st.text_input("smoothness error")
    smoothness_error=str(smoothness_error)

    compactness_error=st.text_input("compactness error")
    compactness_error=str(compactness_error)

    concavity_error=st.text_input("concavity error")
    concavity_error=str(concavity_error)

    concave_points_error=st.text_input("concave points error")
    concave_points_error=str(concave_points_error)

    symmetry_error=st.text_input("symmetry error")
    symmetry_error=str(symmetry_error)

    fractal_dimension_error=st.text_input("fractal dimension error")
    fractal_dimension_error=str(fractal_dimension_error)

    worst_radius=st.text_input("worst radius")
    
    worst_radius=str(worst_radius)

    worst_texture=st.text_input("worst texture")
    worst_texture=str(worst_texture)

    worst_perimeter=st.text_input("worst perimeter")
    worst_perimeter=str(worst_perimeter)

    worst_area=st.text_input("worst area")
    worst_area=str(worst_area)

    worst_smoothness=st.text_input("worst smoothness")
    worst_smoothness=str(worst_smoothness)
    
    worst_compactness=st.text_input("worst compactness")
    worst_compactness=str(worst_compactness)

    worst_concavity=st.text_input("worst concavity")
    worst_concavity=str(worst_concavity)

    worst_concave_points=st.text_input("worst concave points")
    worst_concave_points=str(worst_concave_points)

    worst_symmetry=st.text_input("worst symmetry")
    worst_symmetry=str(worst_symmetry)

    worst_fractal_dimension=st.text_input("worst fractal dimension")
    worst_fractal_dimension=str(worst_fractal_dimension)

    features=[radius,texture,perimeter,area,smootness,compactness,concavity,concave_points,symmetry,fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]
    # print(type(features[0]))
    # s=np.array(features)
    # a=s.astype(float)
    # print(type(a[0]))
    # b=sc.fit_transform([a])
    
    
    
    # a=sc.fit_transform([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.014600,0.02387,0.013150,0.01980,0.002300,15.11,19.26,99.70,711.2,0.1440,0.1773,0.2390,0.12880,0.2977,0.07259]])

    # # a=sc.fit_transform([[12.340,22.22,79.85,464.5,0.10120,0.10150,0.053700,0.028220,0.1551,0.06761,0.2949,1.6560,1.955,21.55,0.011340,0.031750,0.031250,0.011350,0.01879,0.005348,13.58,28.68,87.36,553.0,0.14520,0.23380,0.168800,0.08194,0.2268,0.09082]])

    result=""



    model_choice = st.selectbox("Select Model",["Logistic Regression","KNearest Neighbors","DecisionTree","Random Forest","Naive Baiyes"])
    
           
    if st.button("Predict"):

        if model_choice=='Logistic Regression':
            Logistic_Regression_model=pickle.load(open('Logistic_Breast.pickle','rb'))

            s=np.array([[14.690,13.98,98.22,656.1,0.10310,0.18360,0.145000,0.063000,0.2086,0.07406,0.5462,1.5110,4.795,49.45,0.009976,0.052440,0.052780,0.015800,0.02653,0.005444,16.46,18.34,114.10,809.2,0.13120,0.36350,0.321900,0.11080,0.2827,0.09208]])
            # a=s.astype(float)
            
            features=np.array(features)
            features=features.astype(float)
            result=Logistic_Regression_model.predict([features])
            predict_prob=Logistic_Regression_model.predict_proba([features])
            y_pred=Logistic_Regression_model.predict(X_test)
            a=accuracy_score(y_test,y_pred)
            st.success(result)


        elif model_choice=='KNearest Neighbors':
            KNN_model=pickle.load(open('KNN_Breast.pickle','rb'))
            s=np.array([[23.510,24.27,155.10,1747.0,0.10690,0.12830,0.230800,0.141000,0.1797,0.05506,1.0090,0.9245,6.462,164.10,0.006292,0.019710,0.035820,0.013010,0.01479,0.003118,30.67,30.73,202.40,2906.0,0.15150,0.26780,0.481900,0.20890,0.2593,0.07738]])
            # a=s.astype(float)
            # b=sc.fit_transform(a)
            features=np.array(features)
            features=features.astype(float)
            result=KNN_model.predict([features])
            predict_prob=KNN_model.predict_proba([features])
            y_pred=KNN_model.predict(X_test)
            a=accuracy_score(y_test,y_pred)
            st.success(result)

        elif model_choice=="Naive Baiyes":
            NB_model=pickle.load(open('NB_Breast.pickle','rb'))

            features=np.array(features)
            features=features.astype(float)
            result=NB_model.predict([features])
            predict_prob=NB_model.predict_proba([features])
            y_pred=NB_model.predict(X_test)
            a=accuracy_score(y_test,y_pred)
            

        elif model_choice=='DecisionTree':
            Decision_model=pickle.load(open('Decision_Breast.pickle','rb'))

            s=np.array([[23.510,24.27,155.10,1747.0,0.10690,0.12830,0.230800,0.141000,0.1797,0.05506,1.0090,0.9245,6.462,164.10,0.006292,0.019710,0.035820,0.013010,0.01479,0.003118,30.67,30.73,202.40,2906.0,0.15150,0.26780,0.481900,0.20890,0.2593,0.07738]])
            eatures=np.array(features)
            features=features.astype(float)
            result=Decision_model.predict([features])
            predict_prob=Decision_model.predict_proba([features])
            y_pred=Decision_model.predict(X_test)
            a=accuracy_score(y_test,y_pred)
            

        elif model_choice=='Random Forest':
            Random_model=pickle.load(open('Random_Breast.pickle','rb'))

            s=np.array([[23.510,24.27,155.10,1747.0,0.10690,0.12830,0.230800,0.141000,0.1797,0.05506,1.0090,0.9245,6.462,164.10,0.006292,0.019710,0.035820,0.013010,0.01479,0.003118,30.67,30.73,202.40,2906.0,0.15150,0.26780,0.481900,0.20890,0.2593,0.07738]])
            features=np.array(features)
            features=features.astype(float)
            result=Random_model.predict([features])
            predict_prob=Random_model.predict_proba([features])
            y_pred=Random_model.predict(X_test)
            a=accuracy_score(y_test,y_pred)
            
        elif model_choice=="SVM":
            SVM_model=pickle.load(open('SVM_Breast.pickle','rb'))
            
            features=np.array(features)
            features=features.astype(float)
            result=SVM_model.predict([features])
            predict_prob=SVM_model.predict_proba([features])
            y_pred=SVM_model.predict(X_test)
            a=accuracy_score(y_test,y_pred)
        
        if result==0:
            st.warning("You are suffering from Breast Cancer (Malignant)")
            disease='Breast Cancer'
            disease_add(disease)
            pred_probability_score={"Malignant":predict_prob[0][0]*100,"Benign":predict_prob[0][1]*100}
            st.subheader("Prediction Probability Score using {}".format(model_choice))
            st.json(pred_probability_score)
            accuracy="Accuracy of {} is {}".format(model_choice,a*100)
            st.success(accuracy)

            st.subheader(f"Confusion Matrix of {model_choice}")
            cm=confusion_matrix(y_test,y_pred)
            fig,ax=plt.subplots()
            sns.heatmap(cm,annot=True,ax=ax)
            st.write(fig)

            st.subheader(f"Classification Report of {model_choice}")
            a=classification_report(y_test,y_pred,output_dict=True)
            report=pd.DataFrame(a).transpose()
            st.dataframe(report)

            from sklearn.metrics import precision_score
            st.subheader(f"PRECISION OF {model_choice}")
            precision_score=precision_score(y_test,y_pred)*100
            st.success(precision_score)

            from sklearn.metrics import recall_score
            st.subheader(f"Recall OF {model_choice}")
            recall_score=recall_score(y_test,y_pred)*100
            st.success(recall_score)

            from sklearn.metrics import f1_score
            st.subheader(f"F1 Score of {model_choice}")
            f1_score=f1_score(y_test,y_pred)*100
            st.success(f1_score)

            from sklearn.metrics import roc_auc_score,roc_curve
            st.subheader(f"ROC AUC Score of {model_choice}")
            roc_score=roc_auc_score(y_test,y_pred)*100
            st.success(roc_score)

            st.subheader("Home Remedies/Precautions For Your Disease")
            disease='Breast Cancer'
            precaution_reply=precaution(disease)
            st.success(precaution_reply)



            doctor_subheading_title="""<h3 style="font-size: 2rem;text-align:center;">Go to DOCTOR APPOINTMENT section to book doctor's appointment.</h3>"""
            st.markdown(doctor_subheading_title,unsafe_allow_html=True)



        elif result==1:
            st.success("You are not suffering from Breast Cancer (Benign)")
            pred_probability_score={"Malignant":predict_prob[0][0]*100,"Benign":predict_prob[0][1]*100}
            st.subheader("Prediction Probability Score using {}".format(model_choice))
            st.json(pred_probability_score)
            accuracy="Accuracy of {} is {}".format(model_choice,a*100)
            st.success(accuracy)

            st.subheader(f"Confusion Matrix of {model_choice}")
            cm=confusion_matrix(y_test,y_pred)
            fig,ax=plt.subplots()
            sns.heatmap(cm,annot=True,ax=ax)
            st.write(fig)

            st.subheader(f"Classification Report of {model_choice}")
            a=classification_report(y_test,y_pred,output_dict=True)
            report=pd.DataFrame(a).transpose()
            st.dataframe(report)




elif selected=='DOCTOR APPOINTMENT':
    
    appointment()

elif selected=='Plot':

    dataset_title = '<p style="font-family:Georgia; color:##00FFFF; font-size: 29px;text-align:center;">Breast Cancer Data</p>'
    st.markdown(dataset_title, unsafe_allow_html=True)

    from sklearn.datasets import load_breast_cancer 
    cancer_dataset=load_breast_cancer()   
    cancer_df=pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns=np.append(cancer_dataset['feature_names'],['target']))
    pd.set_option('display.max_columns',None)
    st.dataframe(cancer_df)

    descriptive_statistics_title = '<p style="font-family:Georgia; color:##00FFFF; font-size: 29px;text-align:center;">Descriptive Statistics.</p>'
    st.markdown(descriptive_statistics_title, unsafe_allow_html=True)
    st.dataframe(cancer_df.describe())

    visualization_title = '<p style="font-family:Georgia; color:##00FFFF; font-size: 29px;text-align:center;">Visualization</p>'
    st.markdown(visualization_title, unsafe_allow_html=True)

    count_variable_title="""<h3 style="font-size: 2rem;text-align:center;">Count of Target Variable.</h3>"""
    st.markdown(count_variable_title,unsafe_allow_html=True)
    fig,ax=plt.subplots()

    cancer_df['target'].value_counts().plot(kind='bar')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    heatmap_nonvalues_title="""<h3 style="font-size: 2rem;text-align:center;">Heatmap of Dataset</h3>"""
    st.markdown(heatmap_nonvalues_title,unsafe_allow_html=True)
    fig,ax=plt.subplots()
    sns.heatmap(cancer_df,ax=ax)
    st.write(fig)

    heatmap_corr_title="""<h3 style="font-size: 2rem;text-align:center;">Heatmap of Dataset(corr)</h3>"""
    st.markdown(heatmap_corr_title,unsafe_allow_html=True)
    st.dataframe(cancer_df.corr())


    fig,ax=plt.subplots(figsize=(20,20))
    sns.heatmap(cancer_df.corr(),annot=True,cmap='coolwarm',square=True,vmin=-1,vmax=1,ax=ax)
    st.write(fig)

    
    area_chart_title="""<h3 style="font-size: 2rem;text-align:center;">Area Chart of Features</h3>"""
    st.markdown(area_chart_title,unsafe_allow_html=True)
    all_columns=cancer_df.columns.to_list()
    feat_choices=st.multiselect("Choose a Feature",all_columns)

    new_df=cancer_df[feat_choices]
    st.area_chart(new_df)

    # fig,ax=plt.subplots()
    # sns.pairplot(cancer_df,hue='target')

    # st.write(fig)
    observation_title="""<h3 style="font-size: 2rem;text-align:center;">Observations</h3>"""
    st.markdown(observation_title,unsafe_allow_html=True)
    
    fig,ax=plt.subplots(figsize=(5,3))
    sns.boxplot(data=cancer_df,x='target',y='mean radius',palette="Set1_r",ax=ax)
    st.write(fig)
    st.write("Malignant tumors have larger radius. So we can say that Malignant cancer cells are larger in size than benign cancer cells.")

    fig,ax=plt.subplots(figsize=(5,3))
    sns.boxplot(data=cancer_df,x='target',y='mean texture',palette="flag",ax=ax)
    st.write(fig)
    st.write("The tissue average of malignant cancer cells is larger.")

    fig,ax=plt.subplots(figsize=(5,3))
    sns.boxplot(data=cancer_df,x='target',y='mean perimeter',palette="Oranges_r",ax=ax)
    st.write(fig)
    st.write("Perimeter thickness of malignant cancer cells is greater than that of benign cancer cells")

    fig,ax=plt.subplots(figsize=(5,3))
    sns.boxplot(data=cancer_df,x='target',y='mean area',palette="prism",ax=ax)
    st.write(fig)
    st.write("The area occupied by malignant cancer cells is greater than that of benign cancer cells.")

    fig,ax=plt.subplots()
    fig=px.pie(cancer_df,values='mean radius',names='target',title='Relation')
    st.write(fig)
    st.write("Above figure show Relation between mean radius and target variable")






# -------------------------------------------------Web page section starts---------------------------------------------------------