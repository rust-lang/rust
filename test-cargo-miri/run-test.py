#!/usr/bin/env python3
'''
Test whether cargo-miri works properly.
Assumes the `MIRI_SYSROOT` env var to be set appropriately,
and the working directory to contain the cargo-miri-test project.
'''

import sys, subprocess, os

CGREEN  = '\33[32m'
CBOLD   = '\33[1m'
CEND    = '\33[0m'

def fail(msg):
    print("\nTEST FAIL: {}".format(msg))
    sys.exit(1)

def cargo_miri(cmd):
    args = ["cargo", "miri", cmd, "-q"]
    if 'MIRI_TEST_TARGET' in os.environ:
        args += ["--target", os.environ['MIRI_TEST_TARGET']]
    return args

def test(name, cmd, stdout_ref, stderr_ref, stdin=b'', env={}):
    print("==> Testing {} <==".format(name))
    ## Call `cargo miri`, capture all output
    p_env = os.environ.copy()
    p_env.update(env)
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=p_env,
    )
    (stdout, stderr) = p.communicate(input=stdin)
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
    test("`cargo miri run` (without isolation)",
        cargo_miri("run"),
        "stdout.ref1", "stderr.ref1",
        stdin=b'12\n21\n',
        env={
            'MIRIFLAGS': "-Zmiri-disable-isolation",
            'MIRITESTVAR': "wrongval", # make sure the build.rs value takes precedence
        },
    )
    test("`cargo miri run` (with arguments and target)",
        cargo_miri("run") + ["--bin", "cargo-miri-test", "--", "hello world", '"hello world"'],
        "stdout.ref2", "stderr.ref2"
    )
    test("`cargo miri run` (subcrate)",
        cargo_miri("run") + ["-p", "subcrate"],
        "stdout.ref3", "stderr.ref3",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )

def test_cargo_miri_test():
    # rustdoc is not run on foreign targets
    is_foreign = 'MIRI_TEST_TARGET' in os.environ
    rustdoc_ref = "test.stderr.ref2" if is_foreign else "test.stderr.ref1"

    test("`cargo miri test`",
        cargo_miri("test"),
        "test.stdout.ref1",rustdoc_ref,
        env={'MIRIFLAGS': "-Zmiri-seed=feed"},
    )
    test("`cargo miri test` (with filter)",
        cargo_miri("test") + ["--", "--format=pretty", "le1"],
        "test.stdout.ref2", rustdoc_ref
    )
    test("`cargo miri test` (without isolation)",
        cargo_miri("test") + ["--", "--format=pretty", "num_cpus"],
        "test.stdout.ref3", rustdoc_ref,
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )
    test("`cargo miri test` (test target)",
        cargo_miri("test") + ["--test", "test", "--", "--format=pretty"],
        "test.stdout.ref4", "test.stderr.ref2"
    )
    test("`cargo miri test` (bin target)",
        cargo_miri("test") + ["--bin", "cargo-miri-test", "--", "--format=pretty"],
        "test.stdout.ref5", "test.stderr.ref2"
    )

os.chdir(os.path.dirname(os.path.realpath(__file__)))

target_str = " for target {}".format(os.environ['MIRI_TEST_TARGET']) if 'MIRI_TEST_TARGET' in os.environ else ""
print(CGREEN + CBOLD + "## Running `cargo miri` tests{}".format(target_str) + CEND)

if not 'MIRI_SYSROOT' in os.environ:
    # Make sure we got a working sysroot.
    # (If the sysroot gets built later when output is compared, that leads to test failures.)
    subprocess.run(cargo_miri("setup"), check=True)
test_cargo_miri_run()
test_cargo_miri_test()

print("\nTEST SUCCESSFUL!")
sys.exit(0)
