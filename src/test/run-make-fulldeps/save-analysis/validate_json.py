#!/usr/bin/env python2
from __future__ import print_function

# some sanity checks for save-analysis

# pylint: disable=C,R

import sys
import json
import os.path

out_dir = sys.argv[1]
analysis = json.loads(sys.stdin.readline().strip())

for def_ in analysis['defs']:
    assert def_['id']['krate'] == 0, "defines must have local ids"

defs = {
    d['id']['index']: d for d in analysis['defs']
}

assert 'prelude' in analysis and analysis[
    'prelude'] is not None, "need prelude for sanity checks"

externals = {c['num'] for c in analysis['prelude']['external_crates']}


def check_sane_ids(data, debug_path=None):
    # check that all Ids refer to something plausible, return count
    if debug_path is None:
        debug_path = []

    if isinstance(data, dict):
        if len(data.keys()) == 2 and 'krate' in data and 'index' in data:
            # FIXME: wherever this value exists we should be using Option
            # instead >:(
            if data['krate'] == 0xffffffff or data['index'] == 0xffffffff:
                return 1

            if data['krate'] == 0:
                # FIXME: run this check! it's weird we don't pass it
                # assert data['index'] in defs, "unknown index {} [{}]".format(data[
                #     'index'], debug_path)
                pass
            else:
                assert data['krate'] in externals, "unknown external crate {} [{}]".format(data[
                    'krate'], debug_path)

            return 1
        else:
            # not an index
            count = 0
            for k, v in data.items():
                count += check_sane_ids(v, debug_path + [k])
            return count
    elif isinstance(data, list):
        count = 0
        for i, v in enumerate(data):
            count += check_sane_ids(v, debug_path + [i])
        return count
    else:
        return 0

assert check_sane_ids(
    analysis) > 0, "didn't find anything that looked like an rls_data::Id, did the format change?"

for def_ in analysis['defs']:
    if 'sig' not in def_ or def_['sig'] is None:
        continue

    # write out sigs to check parsing
    with open(os.path.join(out_dir, def_['name'] + '.rs'), 'w') as f:
        if def_['kind'] in ['Function', 'Struct', 'Enum']:
            f.write(def_['sig']['text'])
        if def_['kind'] == 'Field':
            f.write('struct _test {')
            f.write(def_['sig']['text'])
            f.write('}')
        if def_['kind'] in ['TupleVariant', 'StructVariant']:
            f.write('enum _test {')
            f.write(def_['sig']['text'])
            f.write('}')
        if def_['kind'] == 'Method':
            f.write('impl _test {')
            f.write(def_['sig']['text'])
            f.write('}')
