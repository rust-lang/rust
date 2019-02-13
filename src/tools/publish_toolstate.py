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

REPOS = {
    'miri': 'https://github.com/solson/miri',
    'clippy-driver': 'https://github.com/rust-lang/rust-clippy',
    'rls': 'https://github.com/rust-lang/rls',
    'rustfmt': 'https://github.com/rust-lang/rustfmt',
    'book': 'https://github.com/rust-lang/book',
    'nomicon': 'https://github.com/rust-lang-nursery/nomicon',
    'reference': 'https://github.com/rust-lang-nursery/reference',
    'rust-by-example': 'https://github.com/rust-lang/rust-by-example',
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

def issue(
    tool,
    maintainers,
    relevant_pr_number,
    relevant_pr_user,
    pr_reviewer,
):
    # Open an issue about the toolstate failure.
    gh_url = 'https://api.github.com/repos/rust-lang/rust/issues'
    assignees = [x.strip() for x in maintainers.split('@') if x != '']
    assignees.append(relevant_pr_user)
    response = urllib2.urlopen(urllib2.Request(
        gh_url,
        json.dumps({
            'body': textwrap.dedent('''\
            Hello, this is your friendly neighborhood mergebot.
            After merging PR {}, I observed that the tool {} no longer builds.
            A follow-up PR to the repository {} is needed to fix the fallout.

            cc @{}, do you think you would have time to do the follow-up work?
            If so, that would be great!

            cc @{}, the PR reviewer, and @rust-lang/compiler -- nominating for prioritization.

            ''').format(relevant_pr_number, tool, REPOS[tool], relevant_pr_user, pr_reviewer),
            'title': '`{}` no longer builds after {}'.format(tool, relevant_pr_number),
            'assignees': assignees,
            'labels': ['T-compiler', 'I-nominated'],
        }),
        {
            'Authorization': 'token ' + github_token,
            'Content-Type': 'application/json',
        }
    ))
    response.read()

def update_latest(
    current_commit,
    relevant_pr_number,
    relevant_pr_url,
    relevant_pr_user,
    pr_reviewer,
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
            build_failed = False

            for os, s in current_status.items():
                old = status[os]
                new = s.get(tool, old)
                status[os] = new
                if new > old:
                    # things got fixed or at least the status quo improved
                    changed = True
                    message += '🎉 {} on {}: {} → {} (cc {}, @rust-lang/infra).\n' \
                        .format(tool, os, old, new, MAINTAINERS.get(tool))
                elif new < old:
                    # tests or builds are failing and were not failing before
                    changed = True
                    title = '💔 {} on {}: {} → {}' \
                        .format(tool, os, old, new)
                    message += '{} (cc {}, @rust-lang/infra).\n' \
                        .format(title, MAINTAINERS.get(tool))
                    # only create issues for build failures. Other failures can be spurious
                    if new == 'build-fail':
                        build_failed = True

            if build_failed:
                try:
                    issue(
                        tool, MAINTAINERS.get(tool),
                        relevant_pr_number, relevant_pr_user, pr_reviewer,
                    )
                except IOError as (errno, strerror):
                    # network errors will simply end up not creating an issue, but that's better
                    # than failing the entire build job
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise

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

    # assume that PR authors are also owners of the repo where the branch lives
    relevant_pr_match = re.search(
        'Auto merge of #([0-9]+) - ([^:]+):[^,]+ r=([^\s]+)',
        cur_commit_msg,
    )
    if relevant_pr_match:
        number = relevant_pr_match.group(1)
        relevant_pr_user = relevant_pr_match.group(2)
        relevant_pr_number = 'rust-lang/rust#' + number
        relevant_pr_url = 'https://github.com/rust-lang/rust/pull/' + number
        pr_reviewer = relevant_pr_match.group(3)
    else:
        number = '-1'
        relevant_pr_user = '<unknown user>'
        relevant_pr_number = '<unknown PR>'
        relevant_pr_url = '<unknown>'
        pr_reviewer = '<unknown reviewer>'

    message = update_latest(
        cur_commit,
        relevant_pr_number,
        relevant_pr_url,
        relevant_pr_user,
        pr_reviewer,
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
