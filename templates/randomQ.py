import random
qnos=5
questions = ['How was the service ?', 'How can we improve our services ?', 'Would you recommend our products?', 'How was the delivery speed ?', 'Whats your rating ?']
random.shuffle(questions)  #Mixes the items in "questions" into a random order

def returnfn(z):
    if(z==qnos):
        return 'Confirm your reponse for analysis'
    else:
        return str(questions[z])
