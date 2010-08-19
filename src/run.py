#!/usr/bin/env python

import os
import sys
import time
import glob
import fnmatch
from optparse import OptionParser

rustDir = os.path.abspath('.')
rustTestDir = rustDir + "/test";
rustTestRunPassDir = rustTestDir + "/run-pass";
rustTestRunFailDir = rustTestDir + "/run-fail";
rustTestCompileFailDir = rustTestDir + "/run-compile-fail";
rustTestRunBenchDir = rustTestDir + "/run-bench";

parser = OptionParser()
parser.set_usage("run.py [options] pattern : run.py -n 100 \"bas*\" -q");
parser.add_option("-n", dest="repetitions",
                  help="number of repetitions", metavar="NUMBER")
parser.add_option("-q", action="store_true", dest="quiet", default=False,
                  help="suppresses rust log output")
parser.add_option("-l", dest="log", default="",
                  help="rust log")
parser.add_option("-v", action="store_true", dest="valgrind", default=False,
                  help="runs under valgrind")
parser.add_option("-t", action="store_true", dest="terminate", default=False,
                  help="terminate on first failure")
parser.add_option("-p", action="store_true", dest="printSource",
                  default=False, help="prints the test case's source")
parser.add_option("-s", dest="seed", metavar="NUMBER", default=-1,
                  help="seeds the rust scheduler, use -1 to generate seeds, "
                  + " or >= 0 to specify a seed")

(options, args) = parser.parse_args()

def getRustTests(filter):
    tests = []
    for root, dirnames, filenames in os.walk(rustTestDir):
      for filename in fnmatch.filter(filenames, filter + '.rs'):
          tests.append(os.path.join(root, filename).
                       replace(rustDir + "/", ""));
    return tests


if len(args) != 1:
    parser.print_usage();
    sys.exit(0);

tests = getRustTests(args[0]);

# Make
for rustProgram in tests:
    print "Making: " + rustProgram;
    result = os.system("make " + rustProgram.replace(".rs", ".x86")) >> 8;
    if (result != 0):
        print "Make failed!";
        sys.exit(1);

if (options.log != ""):
    os.putenv("RUST_LOG", options.log);

if (options.quiet):
    os.putenv("RUST_LOG", "none");

# Rut
totalPassed = 0;
repetitions = 1;
for rustProgram in tests:
    repetitions = 1;
    if (options.repetitions):
        repetitions = int(options.repetitions);
    passed = 0;
    if (options.printSource):
        os.system("cat " + rustProgram);
    for i in range(0, repetitions):
        print "Running: " + rustProgram + " " + str(i) + \
              " of " + str(repetitions);
        if (options.seed):
            if (int(options.seed) >= 0):
              os.putenv("RUST_SEED", options.seed);
            else:
              os.putenv("RUST_SEED", str(i));
        command = rustProgram.replace(".rs", ".x86");
        if (options.valgrind):
            command = "valgrind --leak-check=full "  + \
                      "--quiet --vex-iropt-level=0 " + \
                      "--suppressions=etc/x86.supp " + \
                      command;
        print "Running Command: " + command;
        result = os.system(command);
        exitStatus = result >> 8;
        signalNumber = result & 0xF;
        if (result == 0):
            passed += 1;
        elif (options.terminate):
            sys.exit(1);
    print "Result for: " + rustProgram + " " + str(passed) + \
          " of " + str(repetitions) + " passed.";
    totalPassed += passed;
print "Total: " + str(totalPassed) + " of " + \
      str(len(tests) * repetitions) + " passed."
