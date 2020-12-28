from flask import Flask, render_template,url_for,request
import app
from templates import randomQ
from textblob import TextBlob

app=Flask(__name__)
app.static_folder = 'static'
msg_inp=[]
num=-1

def update(n):
    global num
    num=n+1

def numIni():
    global num
    num=0
    global msg_inp
    msg_inp=[]

def adder(text):
    global msg_inp
    msg_inp.append(text)

@app.route("/")
def home():
    numIni()
    return render_template("index.html")


@app.route("/res")
def result():
    li=[]
    for i in msg_inp:
        testimonial=TextBlob(i)
        li.append(testimonial.sentiment.polarity)

    senti=[]
    for polar in li:
        if polar > 0.0 and polar <0.5:
            senti.append('mildly positive')
        elif polar > 0.5:
            senti.append('positive')
        elif polar < 0.0 and polar > -0.5:
            senti.append('mildly negative')
        elif polar < -0.5:
            senti.append('negative')
        elif polar == 0:
            senti.append('neutral')
    return render_template("result.html",list=senti)
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    update(num)
    adder(userText)
    return str(randomQ.returnfn(num))

if __name__=="__main__":
    app.run(debug=True)