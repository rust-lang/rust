#!/usr/bin/env python3
'''
Test whether cargo-miri works properly.
Assumes the `MIRI_SYSROOT` env var to be set appropriately,
and the working directory to contain the cargo-miri-test project.
'''

import sys, subprocess

def test_cargo_miri():
    print("==> Testing `cargo miri` <==")
    ## Call `cargo miri`, capture all output
    # FIXME: Disabling validation, still investigating whether there is UB here
    p = subprocess.Popen(
        ["cargo", "miri", "-q", "--", "-Zmiri-disable-validation"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    (stdout, stderr) = p.communicate()
    stdout = stdout.decode("UTF-8")
    stderr = stderr.decode("UTF-8")
    # Show output
    print("=> captured stdout <=")
    print(stdout, end="")
    print("=> captured stderr <=")
    print(stderr, end="")
    # Test for failures
    if p.returncode != 0:
        sys.exit(1)
    if stdout != open('stdout.ref').read():
        print("stdout does not match reference")
        sys.exit(1)
    if stderr != open('stderr.ref').read():
        print("stderr does not match reference")
        sys.exit(1)

def test_cargo_miri_test():
    print("==> Testing `cargo miri test` <==")
    subprocess.check_call(["cargo", "miri", "test"])

test_cargo_miri()
test_cargo_miri_test()
sys.exit(0)
