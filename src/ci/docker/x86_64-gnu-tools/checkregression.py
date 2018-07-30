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

import sys
import json

if __name__ == '__main__':
    os_name = sys.argv[1]
    toolstate_file = sys.argv[2]
    current_state = sys.argv[3]
    verb = sys.argv[4] # 'regressed' or 'changed'

    with open(toolstate_file, 'r') as f:
        toolstate = json.load(f)
    with open(current_state, 'r') as f:
        current = json.load(f)

    regressed = False
    for cur in current:
        tool = cur['tool']
        state = cur[os_name]
        new_state = toolstate.get(tool, '')
        if verb == 'regressed':
            updated = new_state < state
        elif verb == 'changed':
            updated = new_state != state
        else:
            print('Unknown verb {}'.format(updated))
            sys.exit(2)
        if updated:
            print(
                'The state of "{}" has {} from "{}" to "{}"'
                .format(tool, verb, state, new_state)
            )
            regressed = True

    if regressed:
        sys.exit(1)
