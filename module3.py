#load in data
import graphlab
products=graplab.SFrames('/Volumes/Seagate Expansion Drive/learning topic/machine learning/classification and logistic regression/week 2/data/amazon_baby_subset.gl/')

#show up the number of postive/negative reviews
print 'number of positive reviews=',len(products[products['sentiment']==1])
print 'number of negative reviews=',len(products[products['sentiment']==-1])

#apply text cleaning on the review data
#In last one, we used all words in building bag-of-words features, but here we limit ourselves to 193 words.
#We compiled a list of 193 most frequent words into a JSON file.

#First, we load these words from JSON file:
import json
with open('/Volumes/Seagate Expansion Drive/learning topic/machine learning/classification and logistic regression/week 2/data/important_words.json','r') as f: #read the list of most frequent words
    important_words=json.load(f)
important_words=[str(s) for s in important_words]

print important_words

#As it was before, we need to remove the punctuation using python's built-in string functionality
#Compute the counts for important_words
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)
products['review_clean']=products['review'].apply(remove_punctuation)

for word in important_words:
    products[word]=products['review_clean'].apply(lambda s : s.split().count(word))
#Now we get new columns with number of important_words shown in each review


#Convert SFrame to NumPy array
import numpy as np

#We now define a function that extract columns from an SFrame and converts them into NumPy array
#Two arrays are returned: one representing features and another representing class label(sentiment)
#Note: the feature matrix includes an additional column 'intercept' to take account of the intercept term
def get_numpy_data(data_sframe, features, lable):
    data_sframe['intercept']=1
    features=['intercept']+features
    feature_sframe=data_sframe[features]
    feature_matrix=features_sframe.to_numpy()
    label_sarray=data_sframe[label]
    label_array=label_sarray.to_numpy()
    return(feature_matrix,label_array)
feature_matrix,sentiment=get_numpy_data(products, important_words, 'sentiment')


#Estimating conditional probability with link function
#For logistic regression, the linked function is: P(yi=1|xi,W)=1/(1+exp(-WTh(xi)))
#First, we define the function to compute probability
def predict_probability(feature_matrix, coefficients):
    correct_score=np.dot(feature_matrix,coefficients.transpose())
    predictions=1/(1+np.exp(-correct_score))
    return predictions
#To make sure the function is right, we have provided a few examples
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.), 1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_predictions           =', correct_predictions
print 'output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients)

#Next, we want to compute derivative of log likelihood with respect to a single coefficient
#Recall from lecture: fl/fwj=sigma(i to N)(hj(xi)(I[yi=1]-P(yi=1|xi,W)))
def feature_derivative(errors, feature):
    #compute the dot product of errors vector and feature
    derivative=np.dot(errors.transpose(), feature)
    return derivative

#Recall log likelihood function, we have ll(w)=sum(i=1,N)((I(yi=1)-1)WTh(xi)-ln(1+exp(-WTh(xi))))
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator=(sentiment==1)
    scores=np.dot(feature_matrix, coefficients)
    logexp=np.log(1+np.exp(-scores))

    #simple check to prevent overflow
    mask=np.isinf(logexp)
    logexp[exp]=-scores[mask]

    lp=np.sum((indicator-1)*scores-logexp)
    return lp

#check point
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])

correct_indicators  = np.array( [ -1==+1,                                       1==+1 ] )
correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_log_likelihood           =', correct_ll
print 'output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients)

#Gradient Steps
#Gradient ascent function that takes gradient steps towards the optimum
from math import sqrt

def predict_probability(feature_matrix, coefficients):
    correct_score=np.dot(feature_matrix,coefficients.transpose())
    predictions=1/(1+np.exp(-correct_score))
    return predictions

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative          
            derivative = np.dot(errors,feature_matrix[:,j])           
            # add the step size times the derivative to the current coefficient
            coefficients[j]=coefficients[j]+step_size*derivative[j]
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
