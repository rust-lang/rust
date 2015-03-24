#!/usr/bin/env python
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
            if sys.platform in platforms:
                func()
        return inner
    return decorator


@only_on(('linux', 'darwin'))
def check_rlimit_core():
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if soft > 0:
        error_unless_permitted('ALLOW_NONZERO_ULIMIT',
          ("The rust test suite will segfault many rustc's in the debuginfo phase.\n"
           "set ALLOW_NONZERO_ULIMIT to ignore this warning\n"))


def main():
    check_rlimit_core()

if __name__ == '__main__':
    main()
    sys.exit(STATUS)
