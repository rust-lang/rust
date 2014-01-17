#!/usr/bin/env python
# xfail-license

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
    map(summarise, logfiles)
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
