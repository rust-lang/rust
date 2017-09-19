#!/usr/bin/env python
# Script to check the assembly generated
import os, sys
import os.path
from subprocess import Popen, PIPE
import argparse

asm_dir = './asm'

files = set()
verbose = False
extern_crate = None

def arm_triplet(arch) :
    triples = { 'armv7' : 'armv7-unknown-linux-gnueabihf',
                'armv8' : 'aarch64-unknown-linux-gnu' }
    return triples[arch]

class File(object):
    def __init__(self, path_rs):
        self.path_rs = path_rs
        self.path_asm_should = os.path.join(os.path.splitext(path_rs)[0] + ".asm")
        self.path_asm_output = os.path.join(os.path.splitext(path_rs)[0] + "_output.asm")
        self.path_llvmir_output = os.path.join(os.path.splitext(path_rs)[0] + "_ir.ll")
        self.name = os.path.splitext(os.path.basename(path_rs))[0]
        self.feature = self.name.split("_")[1]
        self.arch = self.name.split("_")[0]

        if self.feature == "none":
            self.feature = None

    def __str__(self):
        return  "name: " + self.name + ", path-rs: " + self.path_rs + ", path-asm: " + self.path_asm_should + ', arch: ' + self.arch + ", feature: " + str(self.feature)

    def __hash__(self):
        return hash(self.name)

def find_files():
    for dirpath, dirnames, filenames in os.walk(asm_dir):
        for filename in [f for f in filenames if f.endswith(".rs")]:
            files.add(File(os.path.join(dirpath, filename)))

def call(args):
    if verbose:
        print "command: " + str(args)
    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    lines = p.stdout.readlines()
    if verbose and p.returncode != 0:
        error = p.stderr.readlines()
        print >>sys.stdout, lines
        print >>sys.stderr, "ERROR: %s" % error

def compile_file(file):
    if verbose:
        print "Checking: " + str(file) + "..."

    cargo_args = 'cargo rustc --verbose --release -- -C panic=abort '
    if file.feature:
        cargo_args = cargo_args + '-C target-feature=+{}'.format(file.feature)
    if file.arch == 'armv7' or file.arch == 'armv8':
        cargo_args = cargo_args + '--target={}'.format(arm_triplet(file.arch))
    call(str(cargo_args))

    rustc_args = 'rustc --verbose -C opt-level=3 -C panic="abort" --extern %s=target/release/lib%s.rlib --crate-type lib' % (extern_crate, extern_crate);
    if file.feature:
        rustc_args = rustc_args + ' -C target-feature=+{}'.format(file.feature)
    if file.arch == 'armv7' or file.arch == 'armv8':
        rustc_args = rustc_args + ' --target={}'.format(arm_triplet(file.arch))
    rustc_args_asm = rustc_args + ' --emit asm {} -o {}'.format(file.path_rs, file.path_asm_output)
    call(rustc_args_asm)
    rustc_args_ll = rustc_args + ' --emit llvm-ir {} -o {}'.format(file.path_rs, file.path_llvmir_output)
    call(rustc_args_ll)

    if verbose:
        print "...done!"

def diff_files(rustc_output, asm_snippet):
    with open(rustc_output, 'r') as rustc_output_file:
        rustc_output_lines = rustc_output_file.readlines()

    with open(asm_snippet, 'r') as asm_snippet_file:
        asm_snippet_lines = asm_snippet_file.readlines()

    # remove all empty lines and lines starting with "."
    rustc_output_lines = [l.strip() for l in rustc_output_lines]
    rustc_output_lines = [l for l in rustc_output_lines if not l.startswith(".") and not len(l) == 0]
    asm_snippet_lines = [l.strip() for l in asm_snippet_lines]
    asm_snippet_lines = [l for l in asm_snippet_lines if not l.startswith(".") and not len(l) == 0]

    results_differ = False

    if len(rustc_output_lines) != len(asm_snippet_lines):
        results_differ = True

    for line_is, line_should in zip(rustc_output_lines, asm_snippet_lines):
        if line_is != line_should:
            results_differ = True

    if results_differ:
        print "Error: results differ"
        print "Is:"
        print rustc_output_lines
        print "Should:"
        print asm_snippet_lines
        return False

    return True

def check_file(file):
    compile_file(file)
    return diff_files(file.path_asm_output, file.path_asm_should)

def main():

    parser = argparse.ArgumentParser(description='Checks ASM code')
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-extern-crate', dest='extern_crate', default='stdsimd')
    results = parser.parse_args()

    global verbose
    if results.verbose:
        verbose = True

    global extern_crate
    extern_crate = results.extern_crate

    find_files()

    if verbose:
        for f in files:
            print f
    error = False
    for f in files:
        result = check_file(f)
        if not result:
            error = True

    if error == True:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()
