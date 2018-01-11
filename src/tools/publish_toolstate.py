#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import sys
import re
import json
import copy
import datetime
import collections

# List of people to ping when the status of a tool changed.
MAINTAINERS = {
    'miri': '@oli-obk @RalfJung @eddyb',
    'clippy-driver': '@Manishearth @llogiq @mcarton @oli-obk',
    'rls': '@nrc',
    'rustfmt': '@nrc',
}


def read_current_status(current_commit, path):
    '''Reads build status of `current_commit` from content of `history/*.tsv`
    '''
    with open(path, 'rU') as f:
        for line in f:
            (commit, status) = line.split('\t', 1)
            if commit == current_commit:
                return json.loads(status)
    return {}


def update_latest(current_commit, relevant_pr_number, current_datetime):
    '''Updates `_data/latest.json` to match build result of the given commit.
    '''
    with open('_data/latest.json', 'rb+') as f:
        latest = json.load(f, object_pairs_hook=collections.OrderedDict)

        current_status = {
            os: read_current_status(current_commit, 'history/' + os + '.tsv')
            for os in ['windows', 'linux']
        }

        slug = 'rust-lang/rust'
        message = '📣 Toolstate changed by {}!\n\nTested on commit {}@{}.\n\n' \
            .format(relevant_pr_number, slug, current_commit)
        anything_changed = False
        for status in latest:
            tool = status['tool']
            changed = False

            for os, s in current_status.items():
                old = status[os]
                new = s.get(tool, old)
                status[os] = new
                if new > old:
                    changed = True
                    message += '🎉 {} on {}: {} → {}.\n' \
                        .format(tool, os, old, new)
                elif new < old:
                    changed = True
                    message += '💔 {} on {}: {} → {} (cc {}).\n' \
                        .format(tool, os, old, new, MAINTAINERS[tool])

            if changed:
                status['commit'] = current_commit
                status['datetime'] = current_datetime
                anything_changed = True

        if not anything_changed:
            return ''

        f.seek(0)
        f.truncate(0)
        json.dump(latest, f, indent=4, separators=(',', ': '))
        return message


if __name__ == '__main__':
    cur_commit = sys.argv[1]
    cur_datetime = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    cur_commit_msg = sys.argv[2]
    save_message_to_path = sys.argv[3]

    relevant_pr_match = re.search('#[0-9]+', cur_commit_msg)
    if relevant_pr_match:
        relevant_pr_number = 'rust-lang/rust' + relevant_pr_match.group(0)
    else:
        relevant_pr_number = '<unknown PR>'

    message = update_latest(cur_commit, relevant_pr_number, cur_datetime)
    if message:
        print(message)
        with open(save_message_to_path, 'w') as f:
            f.write(message)
    else:
        print('<Nothing changed>')
