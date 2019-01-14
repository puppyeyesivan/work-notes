Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 18:37:05) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> #import modules
>>> from __future__ import division
>>> import graphlab
>>> import math
>>> import string
>>> #step1: import data
>>> products=graphlab.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/week1/amazon_baby.gl/')
This non-commercial license of GraphLab Create for academic use is assigned to yifan1991@gwmail.gwu.edu and will expire on December 19, 2019.
[INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: /tmp/graphlab_server_1547071923.log
>>> products
Columns:
	name	str
	review	str
	rating	float

Rows: 183531

Data:
+-------------------------------+-------------------------------+--------+
|              name             |             review            | rating |
+-------------------------------+-------------------------------+--------+
|    Planetwise Flannel Wipes   | These flannel wipes are OK... |  3.0   |
|     Planetwise Wipe Pouch     | it came early and was not ... |  5.0   |
| Annas Dream Full Quilt wit... | Very soft and comfortable ... |  5.0   |
| Stop Pacifier Sucking with... | This is a product well wor... |  5.0   |
| Stop Pacifier Sucking with... | All of my kids have cried ... |  5.0   |
| Stop Pacifier Sucking with... | When the Binky Fairy came ... |  5.0   |
| A Tale of Baby's Days with... | Lovely book, it's bound ti... |  4.0   |
| Baby Tracker&reg; - Daily ... | Perfect for new parents. W... |  5.0   |
| Baby Tracker&reg; - Daily ... | A friend of mine pinned th... |  5.0   |
| Baby Tracker&reg; - Daily ... | This has been an easy way ... |  4.0   |
+-------------------------------+-------------------------------+--------+
[183531 rows x 3 columns]
Note: Only the head of the SFrame is printed.
You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
>>> #Step2: data processing
>>> #2.1 remove punctuation
>>> def remove_punctuation(text):
	import string
	return text.translate(None, string.punctuation)

>>> review_without_punctuation=products['review'].apply(remove_punctuation)
>>> #2.2 create new columns in 'products', where count the number of each word in review
>>> products['word_count']=graphlab.text_analytics.count_words(review_without_punctuation)
>>> products
Columns:
	name	str
	review	str
	rating	float
	word_count	dict

Rows: 183531

Data:
+-------------------------------+-------------------------------+--------+
|              name             |             review            | rating |
+-------------------------------+-------------------------------+--------+
|    Planetwise Flannel Wipes   | These flannel wipes are OK... |  3.0   |
|     Planetwise Wipe Pouch     | it came early and was not ... |  5.0   |
| Annas Dream Full Quilt wit... | Very soft and comfortable ... |  5.0   |
| Stop Pacifier Sucking with... | This is a product well wor... |  5.0   |
| Stop Pacifier Sucking with... | All of my kids have cried ... |  5.0   |
| Stop Pacifier Sucking with... | When the Binky Fairy came ... |  5.0   |
| A Tale of Baby's Days with... | Lovely book, it's bound ti... |  4.0   |
| Baby Tracker&reg; - Daily ... | Perfect for new parents. W... |  5.0   |
| Baby Tracker&reg; - Daily ... | A friend of mine pinned th... |  5.0   |
| Baby Tracker&reg; - Daily ... | This has been an easy way ... |  4.0   |
+-------------------------------+-------------------------------+--------+
+-------------------------------+
|           word_count          |
+-------------------------------+
| {'and': 5, 'stink': 1, 'be... |
| {'and': 3, 'love': 1, 'it'... |
| {'and': 2, 'quilt': 1, 'it... |
| {'and': 3, 'ingenious': 1,... |
| {'and': 2, 'all': 2, 'help... |
| {'and': 2, 'this': 2, 'her... |
| {'shop': 1, 'noble': 1, 'i... |
| {'and': 2, 'all': 1, 'righ... |
| {'and': 1, 'fantastic': 1,... |
| {'all': 1, 'standarad': 1,... |
+-------------------------------+
[183531 rows x 4 columns]
Note: Only the head of the SFrame is printed.
You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
>>> #2.3 extract sentiment
>>> #rules: classify into sentiment 1 where rating >3 and -1 where rating <3, ignore rating=3
>>> products_filtered=products[products['rating']!=1]
>>> len(products_filtered)
168348
>>> len(products)
183531
>>> products=products[products['rating']!=1]
>>> len(products)
168348
>>> products['sentiment']=products['rating'].apply(lambda rating:+1 if rating>3 else -1)
>>> #2.4 split data into training and test sets with percentage of 80% data in training and 20% in test set
>>> train_data,test_data=products.random_split(.8,seed=1)
>>> len(train_set)

Traceback (most recent call last):
  File "<pyshell#28>", line 1, in <module>
    len(train_set)
NameError: name 'train_set' is not defined
>>> len(train_data)
134696
>>> len(test_data)
33652
>>> #Step3: Modeling with logistic regression
>>> sentiment_model=graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=None)
Logistic regression:
--------------------------------------------------------
Number of examples          : 134696
Number of classes           : 2
Number of feature columns   : 1
Number of unpacked features : 122450
Number of coefficients    : 122451
Starting L-BFGS
--------------------------------------------------------
+-----------+----------+-----------+--------------+-------------------+
| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy |
+-----------+----------+-----------+--------------+-------------------+
| 1         | 6        | 0.000001  | 1.913466     | 0.833521          |
| 2         | 9        | 5.000000  | 2.496786     | 0.865950          |
| 3         | 10       | 5.000000  | 2.756561     | 0.917978          |
| 4         | 11       | 5.000000  | 3.031621     | 0.319861          |
| 5         | 13       | 1.000000  | 3.445498     | 0.920079          |
| 6         | 14       | 1.000000  | 3.706473     | 0.899284          |
| 10        | 19       | 1.000000  | 4.910900     | 0.955181          |
+-----------+----------+-----------+--------------+-------------------+
TERMINATED: Iteration limit reached.
This model may not be optimal. To improve it, consider increasing `max_iterations`.
>>> 
weights

Traceback (most recent call last):
  File "<pyshell#33>", line 1, in <module>
    weights
NameError: name 'weights' is not defined
>>> weights

Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    weights
NameError: name 'weights' is not defined
>>> weights['value']

Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    weights['value']
NameError: name 'weights' is not defined
>>> num_positive_weight=len(weights[weights['value']>=0])

Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    num_positive_weight=len(weights[weights['value']>=0])
NameError: name 'weights' is not defined
>>> #3.1 check out the coefficient of logistic model
>>> weight=sentiment_model.coefficients
>>> weights.column_names()

Traceback (most recent call last):
  File "<pyshell#39>", line 1, in <module>
    weights.column_names()
NameError: name 'weights' is not defined
>>> weight.column_names()
['name', 'index', 'class', 'value', 'stderr']
>>> weight
Columns:
	name	str
	index	str
	class	int
	value	float
	stderr	float

Rows: 122451

Data:
+-------------+-----------+-------+------------------+--------+
|     name    |   index   | class |      value       | stderr |
+-------------+-----------+-------+------------------+--------+
| (intercept) |    None   |   1   |  0.663268668729  |  None  |
|  word_count |  handles  |   1   |  0.136800231614  |  None  |
|  word_count | stripping |   1   | -0.620866429596  |  None  |
|  word_count |   stink   |   1   | -0.0823591189478 |  None  |
|  word_count |   issues  |   1   |  0.306885541763  |  None  |
|  word_count |   rough   |   1   |  -1.04167367052  |  None  |
|  word_count |    get    |   1   | -0.0878030957118 |  None  |
|  word_count |    they   |   1   | 0.0417507192703  |  None  |
|  word_count |  replace  |   1   | -0.207421683007  |  None  |
|  word_count |     to    |   1   | 0.00736250678332 |  None  |
+-------------+-----------+-------+------------------+--------+
[122451 rows x 5 columns]
Note: Only the head of the SFrame is printed.
You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
>>> num_postive_weight=len(weight[weight['value']>0])
>>> num_postive_weight
88975
>>> num_negative_weight=len(weight[weight['value']<0])
>>> num_negative_weight
33476
>>> #notes:positive weights means words have positive effects on sentiment, negative weights means words have negative effects on sentiment
>>> #3.2 predict the logistic regression
>>> sample_test_data=test_dara[10:13]

Traceback (most recent call last):
  File "<pyshell#48>", line 1, in <module>
    sample_test_data=test_dara[10:13]
NameError: name 'test_dara' is not defined
>>> sample_test_data=test_data[10:13]
>>> sample_test_data
Columns:
	name	str
	review	str
	rating	float
	word_count	dict
	sentiment	int

Rows: 3

Data:
+-------------------------------+-------------------------------+--------+
|              name             |             review            | rating |
+-------------------------------+-------------------------------+--------+
| Baby's First Year Undated ... | A friend bought me this ca... |  5.0   |
| My Kindergarten Year - A K... | I was pleasantly surprised... |  5.0   |
| Cloth Diaper Pins Stainles... | I really thought I was get... |  2.0   |
+-------------------------------+-------------------------------+--------+
+-------------------------------+-----------+
|           word_count          | sentiment |
+-------------------------------+-----------+
| {'and': 1, 'be': 1, 'great... |     1     |
| {'detailed': 1, 'and': 2, ... |     1     |
| {'sadand': 1, 'overpriced'... |     -1    |
+-------------------------------+-----------+
[3 rows x 5 columns]

>>> scores=sentiment_model.predicty(sample_test_data,output_type='margin')

Traceback (most recent call last):
  File "<pyshell#51>", line 1, in <module>
    scores=sentiment_model.predicty(sample_test_data,output_type='margin')
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/toolkits/_model.py", line 635, in __getattribute__
    return object.__getattribute__(self, attr)
AttributeError: 'LogisticClassifier' object has no attribute 'predicty'
>>> scores=sentiment_model.predict(samole_test_data,output_type='margin')

Traceback (most recent call last):
  File "<pyshell#52>", line 1, in <module>
    scores=sentiment_model.predict(samole_test_data,output_type='margin')
NameError: name 'samole_test_data' is not defined
>>> scores=sentiment_model.predict(sample_test_data,output_type='margin')

>>> scores
dtype: float
Rows: 3
[6.007152881848187, 3.772478977341191, -0.19674213639455307]
>>> sentiment_predict=sentiment_mode.predict[sample_test_data,output_type='type']
SyntaxError: invalid syntax
>>> sentiment_predict=sentiment_mode.predict(sample_test_data,output_type='type')

Traceback (most recent call last):
  File "<pyshell#56>", line 1, in <module>
    sentiment_predict=sentiment_mode.predict(sample_test_data,output_type='type')
NameError: name 'sentiment_mode' is not defined
>>> sentiment_predict=sentiment_model.predict(sample_test_data,output_type='type')
[ERROR] graphlab.toolkits._main: Toolkit error: Invalid prediction type name type

Traceback (most recent call last):
  File "<pyshell#57>", line 1, in <module>
    sentiment_predict=sentiment_model.predict(sample_test_data,output_type='type')
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/toolkits/classifier/logistic_classifier.py", line 651, in predict
    missing_value_action=missing_value_action)
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/toolkits/_supervised_learning.py", line 137, in predict
    'supervised_learning_predict', options)
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/toolkits/_main.py", line 89, in run
    raise ToolkitError(str(message))
ToolkitError: Invalid prediction type name type
>>> sentiment_predict=sentiment_model.predict(sample_test_data,output_type='class')
>>> sentiment_predict
dtype: int
Rows: 3
[1, 1, -1]
>>> probability_predict=sentiment_model.predict(sample_test_data,output_type='probability')
>>> probability_predict
dtype: float
Rows: 3
[0.9975449568550027, 0.9775218952433449, 0.4509725081462424]
>>> #Step4: Analysis
>>> #4.1 Get the most positive review
>>> #4.2 Get the accuracy of model(create the function of calculating accuracy)
>>> #4.3 Generate a new model with few 'significant word' in word count
>>> test_data['probability']=sentiment_model.predict(test_data,output_type='class')
>>> test_data.topk('probability',k=20)
Columns:
	name	str
	review	str
	rating	float
	word_count	dict
	sentiment	int
	probability	int

Rows: 20

Data:
+-------------------------------+-------------------------------+--------+
|              name             |             review            | rating |
+-------------------------------+-------------------------------+--------+
| Baby Tracker&reg; - Daily ... | A friend of mine pinned th... |  5.0   |
| Baby Tracker&reg; - Daily ... | This has been an easy way ... |  4.0   |
| Nature's Lullabies First Y... | Space for monthly photos, ... |  5.0   |
| Nature's Lullabies Second ... | I completed a calendar for... |  4.0   |
| Nature's Lullabies Second ... | Wife loves this calender. ... |  5.0   |
|  Lamaze Peekaboo, I Love You  | This book is so worth the ... |  5.0   |
|  Lamaze Peekaboo, I Love You  | we just got this book for ... |  5.0   |
|  Lamaze Peekaboo, I Love You  | My son loved this book as ... |  5.0   |
| SoftPlay Twinkle Twinkle E... |                               |  5.0   |
| SoftPlay Baby's First Clot... | It is so hard to find clot... |  5.0   |
+-------------------------------+-------------------------------+--------+
+-------------------------------+-----------+-------------+
|           word_count          | sentiment | probability |
+-------------------------------+-----------+-------------+
| {'and': 1, 'fantastic': 1,... |     1     |      1      |
| {'all': 1, 'standarad': 1,... |     1     |      1      |
| {'a': 1, 'and': 1, 'what':... |     1     |      1      |
| {'and': 5, 'all': 3, 'have... |     1     |      1      |
| {'and': 2, 'dont': 1, 'jus... |     1     |      1      |
| {'and': 1, 'cute': 1, 'say... |     1     |      1      |
| {'and': 3, 'chew': 1, 'all... |     1     |      1      |
| {'infant': 1, 'being': 1, ... |     1     |      1      |
|               {}              |     1     |      1      |
| {'fantastic': 1, 'old': 1,... |     1     |      1      |
+-------------------------------+-----------+-------------+
[20 rows x 6 columns]
Note: Only the head of the SFrame is printed.
You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
>>> #4.2 Accuracy of classifier
>>> def get_classification_accuracy(model,data,true_labels):
	data['predicted_classification']=model.predict(data,output_type='class')
	count=0
	for i in range(0,len(data)):
		if data[i]['predicted_classification']==data[i][true_lables]:
			count=count+1
		else: count=count
	accuracy=count/len(data)
	return accuracy

>>> get_classification_accuracy(sentiment_model,test_data,'sentiment')

Traceback (most recent call last):
  File "<pyshell#79>", line 1, in <module>
    get_classification_accuracy(sentiment_model,test_data,'sentiment')
  File "<pyshell#78>", line 5, in get_classification_accuracy
    if data[i]['predicted_classification']==data[i][true_lables]:
NameError: global name 'true_lables' is not defined
>>> def get_classification_accuracy(model,data,true_labels):
	data['predicted_classification']=model.predict(data,output_type='class')
	count=0
	for i in range(0,len(data)):
		if data[i]['predicted_classification']==data[i][true_labels]:
			count=count+1
		else: count=count
	accuracy=count/len(data)
	return accuracy

>>> get_classification_accuracy(sentiment_model,test_data,'sentiment')
0.8710626411506003

>>> #4.3 subset of words, few words selected as significant words
>>> significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
>>> train_data['word_count_subset']=train_data['word_count'].dict_trim_by_keys(significant_words,exclude=False)
>>> print train_data[0]['word_count_subset']
{}
>>> train_data['word_count_subset']
dtype: dict
Rows: 134696
[{}, {'love': 1, 'disappointed': 1}, {}, {'well': 1, 'little': 1, 'love': 2, 'product': 2, 'loves': 1}, {'great': 1, 'work': 1, 'loves': 1, 'easy': 1}, {'product': 1, 'great': 1, 'would': 1}, {'able': 1}, {'perfect': 1, 'able': 1, 'would': 1}, {'love': 2, 'well': 1, 'easy': 1}, {'perfect': 1, 'easy': 1, 'would': 1}, {'perfect': 2, 'little': 2, 'would': 2}, {'perfect': 1, 'able': 1, 'would': 1}, {'well': 1, 'would': 1}, {'little': 1, 'love': 1}, {}, {}, {'little': 1, 'well': 1, 'great': 2}, {'disappointed': 1}, {'perfect': 1, 'love': 1}, {'disappointed': 1}, {'able': 1, 'work': 1, 'old': 1, 'loves': 1}, {}, {'perfect': 1, 'would': 1, 'loves': 1}, {'perfect': 1, 'less': 1}, {'car': 1, 'great': 1}, {'great': 1, 'love': 1, 'little': 2}, {'perfect': 1}, {'little': 1, 'loves': 1}, {'perfect': 1, 'great': 1}, {'loves': 1}, {}, {'old': 1, 'love': 1, 'would': 1}, {'even': 1, 'would': 2}, {'perfect': 1, 'little': 1, 'great': 2, 'would': 1, 'loves': 1}, {'great': 1, 'love': 1}, {}, {'little': 1, 'easy': 1}, {'old': 1, 'loves': 1}, {'great': 1, 'loves': 1}, {'love': 2}, {'great': 3}, {'little': 1}, {'old': 1}, {'little': 1, 'easy': 1}, {'perfect': 1}, {'would': 1}, {}, {'love': 1}, {}, {'product': 1}, {'even': 1, 'perfect': 1, 'great': 1, 'little': 2}, {'great': 1, 'love': 1}, {'perfect': 1}, {'great': 1}, {'well': 1, 'old': 1}, {'great': 1, 'well': 1}, {'easy': 1}, {'would': 2}, {'easy': 1}, {'product': 2}, {'love': 1}, {'great': 1, 'love': 1}, {'great': 1}, {'old': 1, 'able': 1}, {}, {'great': 2, 'easy': 1}, {'little': 1}, {}, {'little': 1}, {'great': 2}, {'work': 1}, {}, {}, {}, {'disappointed': 1, 'would': 1}, {}, {'even': 1, 'great': 1, 'love': 1, 'would': 2}, {'product': 2, 'love': 1}, {'great': 1, 'old': 1}, {'easy': 1}, {'old': 1, 'loves': 1, 'easy': 1}, {'perfect': 1}, {'great': 1}, {'perfect': 2, 'love': 1}, {'great': 2, 'love': 3}, {'even': 1, 'little': 1, 'love': 1, 'would': 1, 'work': 1, 'able': 1, 'easy': 1}, {}, {}, {'work': 1, 'well': 1}, {'love': 1, 'easy': 1}, {}, {'great': 2}, {'even': 1, 'great': 1, 'easy': 1}, {'old': 1, 'love': 1, 'would': 1}, {'old': 1, 'well': 1, 'loves': 1}, {}, {'little': 1}, {}, {'well': 1}, {'little': 1}, ... ]
>>> 
