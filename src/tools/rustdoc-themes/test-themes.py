#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from os import listdir
from os.path import isfile, join
import subprocess
import sys

FILES_TO_IGNORE = ['main.css']


def print_err(msg):
    sys.stderr.write('{}\n'.format(msg))


def exec_command(command):
    child = subprocess.Popen(command)
    stdout, stderr = child.communicate()
    return child.returncode


def main(argv):
    if len(argv) < 2:
        print_err("Needs rustdoc binary path")
        return 1
    rustdoc_bin = argv[0]
    themes_folder = argv[1]
    themes = [join(themes_folder, f) for f in listdir(themes_folder)
              if isfile(join(themes_folder, f)) and f not in FILES_TO_IGNORE]
    if len(themes) < 1:
        print_err('No theme found in "{}"...'.format(themes_folder))
        return 1
    args = [rustdoc_bin, '-Z', 'unstable-options', '--theme-checker']
    args.extend(themes)
    return exec_command(args)


if __name__ != '__main__':
    print_err("Needs to be run as main")
    sys.exit(1)
else:
    sys.exit(main(sys.argv[1:]))
