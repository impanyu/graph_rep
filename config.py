import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables_by_name
import numpy as np
# this one lets us draw plots in our notebook
#import matplotlib.pyplot as plt
# and we need to do some work with file paths
import os
import tensorflow.contrib.distributions as tfd
import math
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
import utils



dimensions=2
