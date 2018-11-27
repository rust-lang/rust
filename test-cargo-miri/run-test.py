#!/usr/bin/env python3
'''
Test whether cargo-miri works properly.
Assumes the `MIRI_SYSROOT` env var to be set appropriately,
and the working directory to contain the cargo-miri-test project.
'''

import sys, subprocess

def test(name, cmd, stdout_ref, stderr_ref):
    print("==> Testing `{}` <==".format(name))
    ## Call `cargo miri`, capture all output
    p = subprocess.Popen(
        cmd,
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
    if stdout != open(stdout_ref).read():
        print("stdout does not match reference")
        sys.exit(1)
    if stderr != open(stderr_ref).read():
        print("stderr does not match reference")
        sys.exit(1)

def test_cargo_miri_run():
    test("cargo miri run", ["cargo", "miri", "run", "-q"], "stout.ref", "stderr.ref")

def test_cargo_miri_test():
    # FIXME: validation disabled for now because of https://github.com/rust-lang/rust/issues/54957
    test("cargo miri test", ["cargo", "miri", "test", "-q", "--", "-Zmiri-disable-validation"], "stout.ref", "stderr.ref")

test_cargo_miri_run()
test_cargo_miri_test()
sys.exit(0)
