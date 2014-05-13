#!/usr/bin/env python
#
# Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import glob
import sys

if __name__ == '__main__':
    summaries = []
    def summarise(fname):
        summary = {}
        with open(fname) as fd:
            for line in fd:
                splitline = line.strip().split(' ')
                if len(splitline) == 1:
                    continue
                status = splitline[0]
                test = splitline[-1]
                # track bench runs
                if splitline[1] == 'ns/iter':
                    status = 'bench'
                if not summary.has_key(status):
                    summary[status] = []
                summary[status].append(test)
            summaries.append((fname, summary))
    def count(t):
        return sum(map(lambda (f, s): len(s.get(t, [])), summaries))
    logfiles = sys.argv[1:]
    for files in map(glob.glob, logfiles):
        map(summarise, files)
    ok = count('ok')
    failed = count('failed')
    ignored = count('ignored')
    measured = count('bench')
    print "summary of %d test runs: %d passed; %d failed; %d ignored; %d measured" % \
            (len(logfiles), ok, failed, ignored, measured)
    print ""
    if failed > 0:
        print "failed tests:"
        for f, s in summaries:
            failures = s.get('failed', [])
            if len(failures) > 0:
                print "  %s:" % (f)
            for test in failures:
                print "    %s" % (test)
