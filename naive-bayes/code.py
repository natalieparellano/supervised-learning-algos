
# coding: utf-8

#### Setup

# In[1]:

# built-in python library for interacting with .csv files
import csv
# natural language toolkit is a great python library for natural language processing
import nltk
# nltk.download('all') 
# built-in python library for utility functions that introduce randomness
import random
# built-in python library for measuring time-related things
import time
# regular expressions
import re
# plotting
import matplotlib.pyplot as plt


# In[2]:

def get_raw_data( limit ):
    """
    # Step 1: This reads in the rows from the csv file which look like this:
    0, I'm so sad
    1, Happy!

    where the first row is the label; 0=negative, 1=positive
    and the second row is the body of the tweet
    """
    
    # open the file, which we've placed at /home/vagrant/repos/datasets/clean_twitter_data.csv
    # 'rb' means read-only mode and binary encoding
    f    = open('/home/vagrant/repos/datasets/sms_spam_or_ham.csv', 'rb')
    # let's read in the rows from the csv file
    rows = []
    
    for row in csv.reader( f ):
        rows.append( row )

    rows = rows[1:]
    print 'There are %s rows in the complete dataset' % len( rows )
    random.shuffle( rows )
    # close the file
    f.close()
    
    return rows[:limit]


# In[3]:

# segment the data 
def get_training_and_validation_sets( raw_data ):
    """
    This takes the input dataset, randomly shuffles it to ensure we're
    taking an unbiased sample, and then splits the set of features into
    a training set and a validation set.
    """
    # randomly shuffle the feature sets
    random.shuffle( raw_data )

    # get the number of data points that we have
    count          = len( raw_data )
    # 20% of the set, also called "corpus", should be training, as a rule of thumb, but not gospel.

    # we'll slice this list 20% the way through
    slicing_point  = int( .20 * count )

    # the training set will be the first segment
    training_set   = raw_data[:slicing_point]

    # the validation set will be the second segment
    validation_set = raw_data[slicing_point:]
    return training_set, validation_set


# In[4]:

# setup state for testing out methods 
training_set, validation_set = get_training_and_validation_sets( get_raw_data( 10000 ) )                                   
msgs                         = training_set
msg                          = msgs[0]

print 'training set has size %s' % len( training_set )
print msg


# In[5]:

# spam or ham
ham_msgs  = [ m for m in msgs if m[0] == 'ham' ]
spam_msgs = [ m for m in msgs if m[0] == 'spam' ]
print 'There are %s hams' % len( ham_msgs )
print 'There are %s spams' % len( spam_msgs )


#### Helper methods for feature dictionary 

# In[6]:

def get_length_bucket( msg_length ):
    """
    buckets the msg length into either short / medium / long
    """
    if msg_length < 20:
        return "short"
    elif msg_length < 80:
        return "medium"
    else:
        return "long"
    
print msg
print get_length_bucket( len( msg[1] ) )


# In[7]:

def get_num_caps( msg ):
    """
    Get the number of capital letters in the message.
    """
    return sum( 1 for l in msg if l.isupper() )

print msg
get_num_caps( msg[1] )


# In[8]:

def get_num_nums( msg ):
    """
    Get the number of number characters in the message.
    """
    return sum( 1 for l in msg if re.match( '[0-9]', l) )

print msg
get_num_nums( msg[1] )


# In[9]:

def get_pct_caps( msg ):
    """
    Get the percentage of characters in the message that are capital letters.
    """
    num_caps = get_num_caps( msg )
    return float( num_caps ) / len( msg )

print msg
get_pct_caps( msg[1] )


# In[10]:

def get_pct_nums( msg ):
    """
    Get the percentage of characters in the message that are number characters.
    """
    num_nums = get_num_nums( msg )
    return float( num_nums ) / len( msg )

print msg
get_pct_nums( msg[1] )


# In[11]:

# Printing out the distribution for my sample to see what thresholds make sense
pcts_spam = []
for msg in msgs:
    try: # npa: is there an easier way to handle exceptions like these?
        if msg[0] == 'spam':
            pcts_spam.append( get_pct_nums( msg[1] ) )
    except:
        pass

pcts_ham = []
for msg in msgs:
    try:
        if msg[0] == 'ham':
            pcts_ham.append( get_pct_nums( msg[1] ) )
    except:
        pass        
    
plt.hist( pcts_spam )
plt.hist( pcts_ham )


#### Get the ham and spam words

# In[12]:

# normalize msgs
def valid_word( word ):
    """
    Returns true or false depending if the word should be considered valid.
    """
    return re.match( '[a-z]+', word.lower() ) and len( word ) > 2
    
def normalize_msg( msg ):
    """
    Takes the raw data and standardizes (lower case, tokenize), then only return the valid words.
    """
    sentiment, words = msg
    try:
        words = [ word.lower() for word in nltk.word_tokenize( words ) if valid_word( word ) ]
    except:
        # npa: how to handle unicode issues 
        words = [ word.lower() for word in words.split() if valid_word( word ) ]
    return ( sentiment, words )

print msg
print normalize_msg( msg )


# In[13]:

# get all the words
def get_all_words( training_set ):
    """
    Get all the words (with duplicates) for messages in the dataset.
    """
    all_words = []
    for msg in training_set:
        try:
            sentiment, words = normalize_msg( msg )
            all_words.extend( words )
        except:
            pass
    return all_words

wordlist = get_all_words( training_set )
print len( wordlist )


# In[14]:

# get most common words
def get_common_words( wordlist, threshold=0 ):
    """
    Returns de-duped list of all the words, if the frequency of a word in the non de-duped list is greater than a threshold.
    """
    num_words    = len( wordlist ) 
    wordlist     = nltk.FreqDist( wordlist )
    common_words = []
    for word in wordlist:
        count_word = wordlist.freq( word ) * num_words # npa: better way to select items based on an absolute threshold?
        if count_word > threshold:
            common_words.append( ( word, count_word ) )
    return common_words

common_words = get_common_words( wordlist, 10 )
for tup in common_words[:6]:
    print tup    
len( common_words )


# In[15]:

def score_word_listwise( word, ham_words, spam_words, ham_th, spam_th ):
    """
    For a given word, check how often it appears in 'ham' messages compared to 'spam' messages. 
    Categorize the word depending on the ratio of ham to spam.
    """
    ham_count   = ham_words.count( word )
    spam_count  = spam_words.count( word )
    category    = None
    # since we are working with a subset of common words, this is Ok
    if   ham_count > 0 and spam_count == 0:
        category = 'ham'
    elif spam_count > 0 and ham_count == 0:
        category = 'spam'
    elif ( ham_count / spam_count ) >= ham_th:
        category = 'ham'        
    elif ( spam_count / ham_count ) >= spam_th:
        category = 'spam'
    return category 

# setup state
ham_msgs   = [t for t in training_set if t[0] == 'ham']
spam_msgs  = [t for t in training_set if t[0] == 'spam']
ham_words  = get_all_words( ham_msgs )
spam_words = get_all_words( spam_msgs )

print score_word_listwise( 'prize', ham_words, spam_words, 3, 5 ) # npa: how to choose these thresholds?


# In[16]:

def find_ham_and_spam_words( training_set, common_word_th, ham_th, spam_th ):
    """
    A wrapper for other methods. Given a training set, returns all the words that should be considered 'ham' or 'spam' 
    based on their prevalence in ham or spam messages.
    """
    ham_msgs       = [t for t in training_set if t[0] == 'ham']
    spam_msgs      = [t for t in training_set if t[0] == 'spam']
    all_ham_words  = get_all_words( ham_msgs )
    all_spam_words = get_all_words( spam_msgs )
    all_words      = get_all_words( training_set )
    common_words   = get_common_words( all_words, common_word_th )
    ham_words      = []
    spam_words     = []
    
    for word, freq in common_words:
        category = score_word_listwise( word, all_ham_words, all_spam_words, ham_th, spam_th )
        if   category == 'ham':
            ham_words.append( word )
        elif category == 'spam':
            spam_words.append( word )
    return ham_words, spam_words

ham_words, spam_words = find_ham_and_spam_words( training_set, 10, 3, 3 )
print 'spam words:'
print spam_words


# In[17]:

def get_spam_words( msg, spam_words ):
    """
    Given a message, return the words in that message that are 'spam' words.
    """
    contains  = [ word for word in spam_words if word in msg ]        
    return contains 

get_spam_words( msg, spam_words )


# In[18]:

def get_ham_words( msg, ham_words ):
    """
    Given a message, return the words in that message that are 'ham' words.
    """    
    contains  = [ word for word in ham_words if word in msg ]        
    return contains 

get_ham_words( msg, ham_words )


#### Make dictionary

# In[19]:

# make dict
def msg_features( msg, spam_words, ham_words ):
    """
     Returns a dictionary of the features of the msg we want our model
    to be based on, e.g. msg_length.

    So if the msg was "Hey!", the output of this function would be
    {
        "length": "short"
    }

    If the msg was "Hey this is a really great idea and I think that we should totally implement this technique",
    then the output would be
    {
        "length": "medium"
    }
    """
    msg      = msg.lower()
    sw       = get_spam_words( msg, spam_words )
    
    base_dict  = {
        "length"        : get_length_bucket( len( msg ) ), 
        "contains_excl" : ( "!" in msg ),
        "many_excl"     : msg.count( "!" ) > 2,
        "contains_spam" : len( sw ) > 0,
        "many_spam"     : len( sw ) > 1,
        "contains_?"    : ( "?" in msg ),
        "many_?"        : msg.count( "?" ) > 1,
        "many_nums"     : get_pct_nums( msg ) > 0.1
    }

    return base_dict


#### Model Execution

# In[20]:

# apply dict on data 
def make_feature_set( data_set, spam_words, ham_words ):
    """
    # Step 2: Turn the raw data rows into feature dictionaries using `twitter_features` function above.

    The output of this function run on the example in Step 1 will look like this:
    [
        ({"length": "short"}, 0), # this corresponds to 0, I'm so sad
        ({"length": "short"}, 1) # this corresponds to 1, Happy!
    ]

    You can think about this more abstractly as this:
    [
        (feature_dictionary, label), # corresponding to row 0
        ... # corresponding to row 1 ... n
    ]
    """
    # now let's generate the output that we specified in the comments above
    output_data = []

    # let's just run it on 100,000 rows first, instead of all 1.5 million rows
    # when you experiment with the `twitter_features` function to improve accuracy
    # feel free to get rid of the row limit and just run it on the whole set
    
    for row in data_set:
        try:
            # Remember that row[0] is the label, either 0 or 1
            # and row[1] is the msg body

            # get the label
            label        = row[0]

            # get the msg body and compute the feature dictionary
            feature_dict = msg_features( row[1], spam_words, ham_words )

            # add the tuple of feature_dict, label to output_data
            data         = ( feature_dict, label )

            output_data.append( data )
        except:
#             print 'Failed to make dict for %s' % row
            pass

    return output_data


# In[21]:

# run model and print results 
def run_classification( training_set, validation_set ):
    # train the NaiveBayesClassifier on the training_set
    classifier = nltk.NaiveBayesClassifier.train( training_set )
    # let's see how accurate it was
    accuracy   = nltk.classify.accuracy( classifier, validation_set )
    print "The accuracy was.... {}".format( accuracy )
    return classifier


# In[22]:

# see the outcomes
def see_outcomes( classifier, dataset, bad_words, good_words ):
    outcomes = []
    for data in dataset:
        outcome         = {}
        outcome['data'] = data 
        try:
            sentiment, words = data # npa: this is where it could fail 
            guess            = classifier.classify( msg_features( words, bad_words, good_words ) )
            outcome['was']   = sentiment
            outcome['guess'] = guess
            
            if guess != sentiment:
                outcome['type'] = 'incorrect'
                outcome['desc'] = 'was %s guessed %s' % ( sentiment, guess )
            else:
                outcome['type'] = 'correct'
                outcome['desc'] = 'correct %s' % sentiment 
        except:
            outcome['type']     = 'unable to process'
            outcome['desc']     = 'unable to process'
            pass
        outcomes.append( outcome )
    return outcomes


# In[23]:

# Now let's use the above functions to run our program
start_time = time.time()

# first shuffle and sample the data
training_set, validation_set = get_training_and_validation_sets( get_raw_data( 200000 ) )

# get bad_words and good_words from the validation_set
good_words, bad_words = find_ham_and_spam_words( training_set, 10, 3, 3 )


# In[24]:

print spam_words


# In[25]:

# apply dict translation
our_training_set   = make_feature_set( training_set, spam_words, ham_words )
our_validation_set = make_feature_set( validation_set, spam_words, ham_words )

print "Size of our data set: {}".format( len( training_set ) + len( validation_set ) )
print "Now training the classifier and testing the accuracy..."

# run the model 
classifier      = run_classification( our_training_set, our_validation_set )

end_time        = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format( completion_time )


# In[26]:

classifier.show_most_informative_features() # npa: want to see...e.g., how many data points are covered by each of these features? is there instrumentation for this?


# In[27]:

# npa: is there better instrumentation for this?
outcomes      = see_outcomes( classifier, validation_set, spam_words, ham_words )
outcomes_desc = nltk.FreqDist( [ o['desc'] for o in outcomes ] )
print outcomes_desc.most_common( 99 )


# In[28]:

# apply the model to a single msg 
def predict( classifier, new_msg ): # npa: can we know the "certainty" that this is correct?
    """
    Given a trained classifier and a fresh data point (a msg),
    this will predict its label, either ham or spam.
    """
    return classifier.classify( msg_features( new_msg, spam_words, ham_words ) )

predict( classifier, 'Hey there, how are you? Call me!' )

