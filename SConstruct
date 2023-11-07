import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from steamroller import Environment
import glob

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables(
    ("DATASETS", "", ["/home/tom/afl.zip"]),
("PROMPT", "", ["this is text produced by OCR, creating numerous mistakes. please produce a cleaned version of this text, making any changes that you believe you have a 90% or above certainty about."]))

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
    BUILDERS={
        "PerformOcrPytesseract" : Builder(
            action="python  scripts/perform_ocry_pytesseract.py --output_file ${TARGETS} --input_file ${SOURCES} --grayscale ${GREYSCALE} --denoise ${DENOISE} --binary_threshold ${BINARY_THRESHOLD} --preprocess ${PREPROCESS} --test_segmentation ${TEST_SEGMENTATION} --prompt ${PROMPT}"
        ),
      "InvokeLlama" : Builder( action = "python scripts/interactive.py --input_file ${SOURCES[0]} --output ${TARGETS[0]}")
    }
)

# OK, at this point we have defined all the builders and variables, so it's
# time to specify the actual experimental process, which will involve
# running all combinations of datasets, folds, model types, and parameter values,
# collecting the build artifacts from applying the models to test data in a list.
#
# The basic pattern for invoking a build rule is:
#
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"
#
# Note how variables are specified in each invocation, and their values used to fill
# in the build commands *and* determine output filenames.  It's a very flexible system,
# and there are ways to make it less verbose, but in this case explicit is better than
# implicit.
#
# Note also how the outputs ("targets") from earlier invocation are used as the inputs
# ("sources") to later ones, and how some outputs are also gathered into the "results"
# variable, so they can be summarized together after each experiment runs.
results = []
for dataset_name in env["DATASETS"]:
   files = []
   for prompt in env["PROMPT"]:
       #for name in env["DATASETS"]: 
       results.append(env.PerformOcrPytesseract("work/{}from{}.json".format("afl_zip_records","first100"), "/home/sbacker2/corpora/afl.zip", GREYSCALE = False, DENOISE = False, BINARY_THRESHOLD = False, PREPROCESS = False,TEST_SEGMENTATION = False, PROMPT = "this is text produced by OCR, creating numerous mistakes. please produce a cleaned version of this text, making any changes that you believe you have a 90% or above certainty about."))
	  
 
llm_ocr = []

for result in results:
    llm_ocr.append(env.InvokeLlama("work/{}fromllama.json".format("100afl"), result))





#output = []   
#output.append(env.CombineJson("work/combined_corrected_output.json", llm_ocr))			



