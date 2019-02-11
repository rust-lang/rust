#!/usr/bin/env python3
'''
Test whether cargo-miri works properly.
Assumes the `MIRI_SYSROOT` env var to be set appropriately,
and the working directory to contain the cargo-miri-test project.
'''

import sys, subprocess

def fail(msg):
    print("TEST FAIL: {}".format(msg))
    sys.exit(1)

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
        fail("Non-zero exit status")
    if stdout != open(stdout_ref).read():
        fail("stdout does not match reference")
    if stderr != open(stderr_ref).read():
        fail("stderr does not match reference")

def test_cargo_miri_run():
    test("cargo miri run", ["cargo", "miri", "run", "-q"], "stdout.ref", "stderr.ref")
    test("cargo miri run (with arguments)",
        ["cargo", "miri", "run", "-q", "--", "--", "hello world", '"hello world"'],
        "stdout.ref", "stderr.ref2"
    )

def test_cargo_miri_test():
    test("cargo miri test", ["cargo", "miri", "test", "-q"], "test.stdout.ref", "test.stderr.ref")
    test("cargo miri test (with filter)",
        ["cargo", "miri", "test", "-q", "--", "--", "impl"],
        "test.stdout.ref2", "test.stderr.ref"
    )

test_cargo_miri_run()
test_cargo_miri_test()
print("TEST SUCCESSFUL!")
sys.exit(0)
