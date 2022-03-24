#!/usr/bin/python

import os
import sys

method = sys.argv[1]
for i in range(9):
    os.system("python numerical_integration.py {} {}".format(i,method)) 
