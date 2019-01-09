#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import json
import datetime
import collections
import textwrap
try:
    import urllib2
except ImportError:
    import urllib.request as urllib2

# List of people to ping when the status of a tool changed.
MAINTAINERS = {
    'miri': '@oli-obk @RalfJung @eddyb',
    'clippy-driver': '@Manishearth @llogiq @mcarton @oli-obk',
    'rls': '@nrc @Xanewok',
    'rustfmt': '@nrc',
    'book': '@carols10cents @steveklabnik',
    'nomicon': '@frewsxcv @Gankro',
    'reference': '@steveklabnik @Havvy @matthewjasper @alercah',
    'rust-by-example': '@steveklabnik @marioidival @projektir',
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


def update_latest(
    current_commit,
    relevant_pr_number,
    relevant_pr_url,
    current_datetime
):
    '''Updates `_data/latest.json` to match build result of the given commit.
    '''
    with open('_data/latest.json', 'rb+') as f:
        latest = json.load(f, object_pairs_hook=collections.OrderedDict)

        current_status = {
            os: read_current_status(current_commit, 'history/' + os + '.tsv')
            for os in ['windows', 'linux']
        }

        slug = 'rust-lang/rust'
        message = textwrap.dedent('''\
            📣 Toolstate changed by {}!

            Tested on commit {}@{}.
            Direct link to PR: <{}>

        ''').format(relevant_pr_number, slug, current_commit, relevant_pr_url)
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
                    message += '🎉 {} on {}: {} → {} (cc {}, @rust-lang/infra).\n' \
                        .format(tool, os, old, new, MAINTAINERS.get(tool))
                elif new < old:
                    changed = True
                    message += '💔 {} on {}: {} → {} (cc {}, @rust-lang/infra).\n' \
                        .format(tool, os, old, new, MAINTAINERS.get(tool))

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
    github_token = sys.argv[4]

    relevant_pr_match = re.search('#([0-9]+)', cur_commit_msg)
    if relevant_pr_match:
        number = relevant_pr_match.group(1)
        relevant_pr_number = 'rust-lang/rust#' + number
        relevant_pr_url = 'https://github.com/rust-lang/rust/pull/' + number
    else:
        number = '-1'
        relevant_pr_number = '<unknown PR>'
        relevant_pr_url = '<unknown>'

    message = update_latest(
        cur_commit,
        relevant_pr_number,
        relevant_pr_url,
        cur_datetime
    )
    if not message:
        print('<Nothing changed>')
        sys.exit(0)

    print(message)

    if not github_token:
        print('Dry run only, not committing anything')
        sys.exit(0)

    with open(save_message_to_path, 'w') as f:
        f.write(message)

    # Write the toolstate comment on the PR as well.
    gh_url = 'https://api.github.com/repos/rust-lang/rust/issues/{}/comments' \
        .format(number)
    response = urllib2.urlopen(urllib2.Request(
        gh_url,
        json.dumps({'body': message}),
        {
            'Authorization': 'token ' + github_token,
            'Content-Type': 'application/json',
        }
    ))
    response.read()
