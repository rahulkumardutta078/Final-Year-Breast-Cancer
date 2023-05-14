import mysql.connector
from datetime import datetime
# mydb=mysql.connector.connect(host='localhost',user='root',password='Rahul@12345',database='disease')
mydb=mysql.connector.connect(host='localhost',user='root',password='Rahul@12345',port=3310,database='disease')
cur=mydb.cursor()

#Functions

# def patient_signup():
# s="create table if not exists patientsignup(Patient_id INT primary key auto_increment,Name varchar(50) not null,Email varchar(50) not null,Password varchar(50) not null,Number INT not null)"
# cur.execute(s)

def add_patient(Name,Email,Password,Number):
    cur.execute('INSERT INTO patientsignup(Name,Email,Password,Number) VALUES(%s,%s,%s,%s)',(Name,Email,Password,Number))
    mydb.commit()

def patient_login(Email,Password):
    cur.execute('SELECT * FROM patientsignup WHERE Email= %s and Password= %s',(Email,Password))
    data = cur.fetchall()
    return data

def current_patient(email):
    cur.execute('select Name from patientsignup where Email=email')
    data=cur.fetchall()
    return data[0][0]

def doctor_appointment(Disease):
    # print(type(disease))
    cur.execute("select Name,Email,Contact,Specialization_In_Disease,City from doctorsignup where Specialization_In_Disease=%s",[Disease])
    data=cur.fetchall()
    if len(data)==0:
        return 0
    else:
        return data


# def doctor_signup():
#     s="create table if not exists doctorsignup(Doctor_id INT primary key auto_increment,Name varchar(50) not null,Email varchar(50) not null,Contact INT not null,Education varchar(50) not null,Year_Of_Exp INT not null,Specialization_In_Disease varchar(50) not null,Gender varchar(50) not null,City varchar(50) not null,Password varchar(50) not null)"
#     cur.execute(s)    

# doctor_signup()

def add_doctor(name,email,contact,education,yr_of_exp,specialization,gender,city,password):
    cur.execute('INSERT INTO doctorsignup(Name,Email,Contact,Education,Year_of_Exp,Specialization_In_Disease,Gender,City,Password) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',(name,email,contact,education,yr_of_exp,specialization,gender,city,password))
    mydb.commit()

def doctor_login(Email,Password):
    cur.execute('SELECT * FROM doctorsignup WHERE Email= %s and Password= %s',(Email,Password))
    data = cur.fetchall()
    return data

def current_doctor(email):
    cur.execute('select Name from doctorsignup where Email=email')
    data=cur.fetchall()
    return data[0][0]

def doctor_available_for_disease(Disease):
    # print(type(disease))
    cur.execute("select Name from doctorsignup where Specialization_In_Disease=%s",[Disease])
    data=cur.fetchall()
    return data

# def disease():
#     s="create table if not exists diseasename(Disease_Id INT primary key auto_increment,Disease_Name varchar(50) not null)"
#     cur.execute(s)


def disease_add(disease):
    cur.execute('INSERT INTO diseasename (Disease_Name) VALUES(%s)',([disease]))
    mydb.commit()


def disease_return():
    cur.execute("SELECT Disease_Name FROM diseasename ORDER BY Disease_Id DESC limit 1")
    data=cur.fetchall()
    
    return data[0][0]

    
    

def alterdiseasename():
    # cur.execute("DELETE FROM diseasename WHERE Disease_Name=%s",[Disease_Name])
    cur.execute("DELETE FROM diseasename where Disease_Id not in (select Disease_Id from (select * from diseasename  order by Disease_Id limit 1) as a)")
    mydb.commit()

def add_appointment1(access_name):
    cur.execute('select Name from patientsignup where Email=%s',([access_name]))
    # Patient_Number=cur.execute('select Number from patientsignup where Name=%s',([access_name]))
    # Doctor_Email=cur.execute('select Email from doctorsignup where Name=%s',([Doctor_Name]))
    # Doctor_Number=cur.execute('select Number from doctorsignup where Name=%s',([Doctor_Name]))
    # Your_Disease=cur.execute('select Specialization_In_Disease from doctorsignup where Name=%s',([Doctor_Name]))
    # date=datetime.now()
    
    data=cur.fetchall()
    return data[0][0]

    # cur.execute('Insert into bookappointment(Patient_Name,Patient_Email,Patient_Number,Doctor_Name,Doctor_Email,Doctor_Number,Your_Disease,Date_of_Appointment) values(%s,%s,%s,%s,%s,%s,%s,%s)',(access_name,Patient_Email,Patient_Number,Doctor_Name,Doctor_Email,Doctor_Number,Your_Disease,date))
    # mydb.commit()


# INSERT INTO classroom(teacher_id,student_id)
#  VALUES ((SELECT id FROM students WHERE s_name='sam'),
#  (SELECT id FROM teacher WHERE t_name='david'));




def add_appointment2(access_name):
    cur.execute('select Number from patientsignup where Email=%s',([access_name]))
    data=cur.fetchall()
    return data[0][0]


def add_appointment3(Doctor_Name):
    cur.execute('select Email from doctorsignup where Name=%s',([Doctor_Name]))
    data=cur.fetchall()
    return data[0][0]

def add_appointment4(Doctor_Name):
    cur.execute('select Contact from doctorsignup where Name=%s',([Doctor_Name]))
    data=cur.fetchall()
    return data[0][0]

# def add_appointment3(Disease_Name):
#     cur.execute('select  from doctorsignup where Name=%s',([Doctor_Name]))
#     data=cur.fetchall()
#     return data[0][0]

def verifyemail(mail):
    cur.execute("SELECT Number FROM patientsignup where Email=%s",([mail]))
    data=cur.fetchall()

    if len(data)==0:
        return 0
    else:
        return data[0][0]

def mailverify(phonecome):
    cur.execute("SELECT Email FROM patientsignup where Number=%s",([phonecome]))
    data=cur.fetchall()
    
    return data[0][0]

def book_app(p1,mail,p2,Doctor_Name,d1,d2,Disease_Name,t):
    cur.execute('Insert into bookappoint(Patient_Name,Patient_Email,Patient_Contact,Doctor_Name,Doctor_Email,Doctor_Contact,Your_Disease,Date_of_Appointment) values(%s,%s,%s,%s,%s,%s,%s,%s)',(p1,mail,p2,Doctor_Name,d1,d2,Disease_Name,t))
    mydb.commit()


# s="create table if not exists bookappoint(Serial_No INT primary key auto_increment,Patient_Name varchar(50) not null,Patient_Email varchar(50) not null,Patient_Contact varchar(50) not null,Doctor_Name varchar(50) not null,Doctor_Email varchar(50) not null,Doctor_Contact varchar(50) not null,Your_Disease varchar(50) not null,Date_of_Appointment varchar(50) not null)"
# cur.execute(s)


def profile_section(z):
    cur.execute("SELECT Patient_Name,Patient_Email,Patient_Contact,Doctor_Name,Doctor_Email,Doctor_Contact,Your_Disease,Date_of_Appointment FROM bookappoint where Patient_Name=%s",([z]))
    data=cur.fetchall()
    if len(data)==0:
        return 0
    else:
    
        return data

def total_available_doctors():
    cur.execute("Select Name,Email,Contact,Education,Year_Of_Exp,Specialization_In_Disease,Gender,City from doctorsignup where Doctor_id!=5")
    data=cur.fetchall()
    return data


# def doctor(Disease_Name):
#     cur.execute("select Name,Email,Contact,Specialization_In_Disease,City from doctorsignup where Specialization_In_Disease=%s",[Disease_Name])
#     data=cur.fetchall()
#     if data==0:
#         return 0
#     else:
#         return 1


def profile_section_doctor(z):
    cur.execute("SELECT Patient_Name,Patient_Email,Patient_Contact,Your_Disease,Date_of_Appointment FROM bookappoint where Doctor_Name=%s",([z]))
    data=cur.fetchall()
    if len(data)==0:
        return 0
    else:
    
        return data