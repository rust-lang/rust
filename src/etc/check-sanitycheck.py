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

import os
import sys
import functools
import resource

STATUS = 0


def error_unless_permitted(env_var, message):
    global STATUS
    if not os.getenv(env_var):
        sys.stderr.write(message)
        STATUS = 1


def only_on(platforms):
    def decorator(func):
        @functools.wraps(func)
        def inner():
            if any(map(lambda x: sys.platform.startswith(x), platforms)):
                func()
        return inner
    return decorator


@only_on(('linux', 'darwin', 'freebsd', 'openbsd'))
def check_rlimit_core():
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if soft > 0:
        error_unless_permitted('ALLOW_NONZERO_RLIMIT_CORE',
          ("The rust test suite will segfault many rustc's in the debuginfo phase.\n"
           "set ALLOW_NONZERO_ULIMIT to ignore this warning\n"))


def main():
    check_rlimit_core()

if __name__ == '__main__':
    main()
    sys.exit(STATUS)
