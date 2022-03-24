#!/usr/bin/python

import os
import sys

method = sys.argv[2]

for i in range(sys.argv[1]):
    os.system("python numerical_integration.py {} {}".format(i,method)) 
