# from d import *
# def g():
#     n=h()
#     print(n)
    
# if __name__ == "__main__":
#     g()
    
    

# from d import *
# def y(a):
#     global z
#     z=a
#     print(z)
#     return z

# import d
# def a():
#     s=d.a
    
#     print(s)


# if __name__ == "__main__":
#     a()



#!/usr/bin/env python

#importing required modules.
import smtplib
import random

#creating function to generate otp.
# def otp():
#     otp_numbers = random.randint(10000,99999)
#     return otp_numbers

#creating function to send email.
def Email(email,message,contact,name):
    
    #establishing connection with email host
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    
    #enter email id and password
    server.login("rahulkumardutta078@gmail.com", "eaxeayyoderfvrbb")

    # enter sender's email id , receiver's email id and message with subject
    server.sendmail("rahulkumardutta078@gmail.com",email, f"""({message}) is the  Query from {email} ,{name}""")
    
    #commiting process.
    server.quit()
    

#calling function 
# email()

def emailappointment(mail,Doctor_Name):
    #establishing connection with email host
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    
    #enter email id and password
    server.login("rahulkumardutta078@gmail.com", "eaxeayyoderfvrbb")

    # enter sender's email id , receiver's email id and message with subject
    server.sendmail("rahulkumardutta078@gmail.com",mail, f"""Your appointmet for doctor {Doctor_Name} is successfully booked.Thank you for choosing us, Doctor will reach you soon via phonecall or videocall.""")
    
    #commiting process.
    server.quit()