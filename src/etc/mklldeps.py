# xfail-license

import os
import sys
import subprocess

components = sys.argv[1].split(' ')

print """// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// WARNING: THIS IS A GENERATED FILE, DO NOT MODIFY
//          take a look at src/etc/mklldeps.py if you're interested
"""

for llconfig in sys.argv[2:]:
    print

    proc = subprocess.Popen([llconfig, '--host-target'], stdout = subprocess.PIPE)
    out, err = proc.communicate()
    arch, os = out.split('-', 1)
    arch = 'x86' if arch == 'i686' or arch == 'i386' else arch
    if 'darwin' in os:
        os = 'macos'
    elif 'linux' in os:
        os = 'linux'
    elif 'freebsd' in os:
        os = 'freebsd'
    elif 'android' in os:
        os = 'android'
    elif 'win' in os or 'mingw' in os:
        os = 'win32'
    cfg = [
        "target_arch = \"" + arch + "\"",
        "target_os = \"" + os + "\"",
    ]

    print "#[cfg(" + ', '.join(cfg) + ")]"

    args = [llconfig, '--libs']
    args.extend(components)
    proc = subprocess.Popen(args, stdout = subprocess.PIPE)
    out, err = proc.communicate()

    for lib in out.strip().split(' '):
        lib = lib[2:] # chop of the leading '-l'
        print "#[link(name = \"" + lib + "\", kind = \"static\")]"
    if os == 'win32':
        print "#[link(name = \"pthread\")]"
        print "#[link(name = \"imagehlp\")]"
    print "extern {}"
