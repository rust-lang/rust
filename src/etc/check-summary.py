#!/usr/bin/env python

import sys

if __name__ == '__main__':
    summaries = []
    def summarise(fname):
        summary = {}
        fd = open(fname)
        for line in fd:
            status, test = line.strip().split(' ', 1)
            if not summary.has_key(status):
                summary[status] = []
            summary[status].append(test)
        summaries.append((fname, summary))
    def count(t):
        return sum(map(lambda (f, s): len(s.get(t, [])), summaries))
    logfiles = sys.argv[1:]
    map(summarise, logfiles)
    ok = count('ok')
    failed = count('failed')
    ignored = count('ignored')
    print "summary of %d test logs: %d passed; %d failed; %d ignored" % \
            (len(logfiles), ok, failed, ignored)
    if failed > 0:
        print "failed tests:"
        for f, s in summaries:
            failures = s.get('failed', [])
            if len(failures) > 0:
                print "  %s:" % (f)
            for test in failures:
                print "    %s" % (test)
