Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 18:37:05) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> #import modules
>>> from __future__ import division
>>> import graphlab
>>> import math
>>> import string
>>> #step1: import data
>>> products=graphlab.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl/')
This non-commercial license of GraphLab Create for academic use is assigned to yifan1991@gwmail.gwu.edu and will expire on December 19, 2019.
[INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: /tmp/graphlab_server_1547071126.log

Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    products=graphlab.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl/')
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/data_structures/sframe.py", line 953, in __init__
    raise ValueError('Unknown input type: ' + format)
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/cython/context.py", line 49, in __exit__
    raise exc_type(exc_value)
IOError: /Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl not found.: unspecified iostream_category error: unspecified iostream_category error
>>> products=gl.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl/')

Traceback (most recent call last):
  File "<pyshell#7>", line 1, in <module>
    products=gl.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl/')
NameError: name 'gl' is not defined
>>> products=graphlab.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl/')

Traceback (most recent call last):
  File "<pyshell#8>", line 1, in <module>
    products=graphlab.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl/')
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/data_structures/sframe.py", line 953, in __init__
    raise ValueError('Unknown input type: ' + format)
  File "/Users/yifanyu/.local/lib/python2.7/site-packages/graphlab/cython/context.py", line 49, in __exit__
    raise exc_type(exc_value)
IOError: /Users/yifanyu/Documents/Topic Learning/ML Classification/amazon_baby.gl not found.: unspecified iostream_category error: unspecified iostream_category error
>>> products

Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    products
NameError: name 'products' is not defined
>>> products=graphlab.SFrame('/Users/yifanyu/Documents/Topic Learning/ML Classification/week1/amazon_baby.gl/')
>>> prodycts

Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    prodycts
NameError: name 'prodycts' is not defined
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
>>> 
