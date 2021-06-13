from flask import Flask, render_template,redirect,url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import Form, StringField , PasswordField , BooleanField
from wtforms import validators
from wtforms.widgets import TextArea
from wtforms.validators import InputRequired , Email, Length
from flask_sqlalchemy import SQLAlchemy
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re


app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'sanket'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SQLALCHEMY_BINDS'] = {'sawaal' : 'sqlite:///sawaal.db' , 'jawaab' : 'sqlite:///jawaab.db'}

db = SQLAlchemy(app)

class PostForm(FlaskForm):
    question = StringField('Question',validators=[] )

class PostForm2(FlaskForm):
    answer   = StringField('Answer', validators=[])
    qnumber  = StringField('For Question no:')

class MyClass():
	def __init__(self, uname, upass, uemail, utext):
		self.uname = uname
		self.upass = upass
		self.uemail = uemail
		self.utext = utext

# class User(db.Model):
# 	id = db.Column(db.Integer, primary_key=True)	
# 	username = db.Column(db.String(15), unique=True)
# 	email = db.Column(db.String(50), unique=True)
# 	password = db.Column(db.String(80))
# 	inp = db.Column(db.String(100))



# class sawaal(db.Model):
# 	__bind_key__ = 'sawaal'
# 	question = db.Column(db.String(200) , primary_key=True)
	
# class jawaab(db.Model):
# 	__bind_key__ = 'jawaab'
# 	id = db.Column(db.Integer, primary_key=True)
# 	answer = db.Column(db.String(200) )    
# 	qnumber  = db.Column(db.Integer())	

class LoginForm(FlaskForm):
	# username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
	# password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
	inputtext = StringField('Enter Text', validators=[InputRequired()])

class SignUpForm(FlaskForm):
	email    = StringField('Email' , validators=[InputRequired() , Email(message='Invalid Email'), Length(max=50) ],default=u'Sanket@sanket.com')
	username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)],default=u'')
	password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)],default=u'')
	inputtext = StringField('Input Text here', validators=[InputRequired(), Length(min=4, max=15)],default=u'')

# class RegistrationForm(Form):
#     username = StringField('Username', [validators.Length(min=4, max=25)])
#     email = StringField('Email Address', [validators.Length(min=6, max=35)])
#     password = PasswordField('New Password', [
#         validators.DataRequired(),
#         validators.EqualTo('confirm', message='Passwords must match')])
#     confirm = PasswordField('Repeat Password')
#     accept_tos = BooleanField('I accept the TOS', [validators.DataRequired()])

@app.route('/')
def Home():
	return render_template('Home.html')

@app.route('/Login'  , methods=['GET' , 'POST'])
def Login():
	form = LoginForm()
	print(100*'-')
	print(form.username.data)
	print(form.inputtext.data)
	if form.is_submitted():
		print("Submitted text is\n",100*'*')
		print(form.inputtext.data)
		# user = MyClass.query.filter_by(uname=form.username.data).first()
		# if user:
		# 	if user.password == form.password.data:
		# 	  	return redirect(url_for('Feed'))
		# return '<h1>Invalid username or password</h1>'

	return render_template('Login.html' , form=form, textarea='fthdrtfjgmjuyg')

# @app.route('/Signup', methods=['GET', 'POST'])
# def Signup():
#     form = RegistrationForm(request.form)
#     if request.method == 'POST' and form.validate():
#         user = User(form.username.data, form.email.data,
#                     form.password.data)
#         #db_session.add(user)
#         flash('Thanks for registering')
#         return redirect(url_for('Login'))
#     return render_template('Signup.html', form=form)


@app.route('/Signup', methods=['GET' , 'POST'])
def Signup():
	form = SignUpForm()
	# print(form.inputtext.data)
	#print(len(form.email.data))
	if form.is_submitted():
		new_user = MyClass(form.username.data,form.email.data, form.password.data, form.inputtext.data)
		inp=form.inputtext.data
		# db.session.add(new_user)	
		# db.session.commit()
		print(inp,20*'-')
		return redirect(url_for('Signup'))

	return render_template('Signup.html' , form=form, textarea=predict_sentence(inp))

@app.route('/Feed', methods=['GET' , 'POST'])
def Feed():
	form = PostForm()
	form2 = PostForm2()

	if form2.validate_on_submit():
		new_answer = jawaab(answer = form2.answer.data , qnumber = form2.qnumber.data)
		if form2.answer.data != '':
			db.session.add(new_answer)	
			db.session.commit()

	if form.validate_on_submit():
		new_question = sawaal(question = form.question.data)
		if form.question.data != '':
			db.session.add(new_question)	
			db.session.commit()

	new_answers = jawaab.query.all()
	new_questions = sawaal.query.all()

	return render_template('Feed.html' , form=form , form2=form2 ,  new_questions=new_questions , new_answers=new_answers )
def convert_to_int(data, data_int):
    """Converts all our text to integers
    :param data: The text to be converted
    :return: All sentences in ints
    """
    all_items = []
    for sentence in data: 
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])
    
    return all_items

def process_sentence(sentence):
    '''Removes all special characters from sentence. It will also strip out
    extra whitespace and makes the string lowercase.
    '''
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,|]', '', sentence.lower().strip())


def predict_sentence(sentence):
    """Converts the text and sends it to the model for classification
    :param sentence: The text to predict
    :return: string - The language of the sentence
    """
    if len(sentence)<75:
        print('Minimum length of string required is 75')
        return ''
    else:
        
        # Clean the sentence
        sentence = process_sentence(sentence)
        
        # Transform and pad it before using the model to predict
        x = np.array(convert_to_int([sentence], vocabint))
        x = sequence.pad_sequences(x, maxlen=100)
        model = model_from_json('D:\\GAIP\\data\\cleaned\\model_json.json')
        prediction = model.predict(x)
        # Get the highest prediction
        lang_index = np.argmax(prediction)
        return intlang[lang_index]


if __name__ == '__main__':
	app.run(debug=True)
