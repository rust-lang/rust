#!/usr/bin/env python3
'''
Test whether cargo-miri works properly.
Assumes the `MIRI_SYSROOT` env var to be set appropriately,
and the working directory to contain the cargo-miri-test project.
'''

import sys, subprocess, os, re

CGREEN  = '\33[32m'
CBOLD   = '\33[1m'
CEND    = '\33[0m'

def fail(msg):
    print("\nTEST FAIL: {}".format(msg))
    sys.exit(1)

def cargo_miri(cmd, quiet = True):
    args = ["cargo", "miri", cmd]
    if quiet:
        args += ["-q"]
    if 'MIRI_TEST_TARGET' in os.environ:
        args += ["--target", os.environ['MIRI_TEST_TARGET']]
    return args

def normalize_stdout(str):
    str = str.replace("src\\", "src/") # normalize paths across platforms
    return re.sub("finished in \d+\.\d\ds", "finished in $TIME", str)

def test(name, cmd, stdout_ref, stderr_ref, stdin=b'', env={}):
    print("Testing {}...".format(name))
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
    if p.returncode == 0 and normalize_stdout(stdout) == open(stdout_ref).read() and stderr == open(stderr_ref).read():
        # All good!
        return
    # Show output
    print("--- BEGIN stdout ---")
    print(stdout, end="")
    print("--- END stdout ---")
    print("--- BEGIN stderr ---")
    print(stderr, end="")
    print("--- END stderr ---")
    fail("exit code was {}".format(p.returncode))

def test_no_rebuild(name, cmd, env={}):
    print("Testing {}...".format(name))
    p_env = os.environ.copy()
    p_env.update(env)
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=p_env,
    )
    (stdout, stderr) = p.communicate()
    stdout = stdout.decode("UTF-8")
    stderr = stderr.decode("UTF-8")
    if p.returncode != 0:
        fail("rebuild failed");
    # Also check for 'Running' as a sanity check.
    if stderr.count(" Compiling ") > 0 or stderr.count(" Running ") == 0:
        print("--- BEGIN stderr ---")
        print(stderr, end="")
        print("--- END stderr ---")
        fail("Something was being rebuilt when it should not be (or we got no output)");

def test_cargo_miri_run():
    test("`cargo miri run` (no isolation)",
        cargo_miri("run"),
        "run.default.stdout.ref", "run.default.stderr.ref",
        stdin=b'12\n21\n',
        env={
            'MIRIFLAGS': "-Zmiri-disable-isolation",
            'MIRITESTVAR': "wrongval", # make sure the build.rs value takes precedence
        },
    )
    # Special test: run it again *without* `-q` to make sure nothing is being rebuilt (Miri issue #1722)
    test_no_rebuild("`cargo miri run` (no rebuild)",
        cargo_miri("run", quiet=False) + ["--", ""],
        env={'MIRITESTVAR': "wrongval"}, # changing the env var causes a rebuild (re-runs build.rs),
                                         # so keep it set
    )
    test("`cargo miri run` (with arguments and target)",
        cargo_miri("run") + ["--bin", "cargo-miri-test", "--", "hello world", '"hello world"'],
        "run.args.stdout.ref", "run.args.stderr.ref",
    )
    test("`cargo miri run` (subcrate, no ioslation)",
        cargo_miri("run") + ["-p", "subcrate"],
        "run.subcrate.stdout.ref", "run.subcrate.stderr.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )

def test_cargo_miri_test():
    # rustdoc is not run on foreign targets
    is_foreign = 'MIRI_TEST_TARGET' in os.environ
    default_ref = "test.cross-target.stdout.ref" if is_foreign else "test.default.stdout.ref"
    filter_ref = "test.filter.cross-target.stdout.ref" if is_foreign else "test.filter.stdout.ref"

    test("`cargo miri test`",
        cargo_miri("test"),
        default_ref, "test.stderr-empty.ref",
        env={'MIRIFLAGS': "-Zmiri-seed=feed"},
    )
    test("`cargo miri test` (no isolation, no doctests)",
        cargo_miri("test") + ["--bins", "--tests"], # no `--lib`, we disabled that in `Cargo.toml`
        "test.cross-target.stdout.ref", "test.stderr-empty.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )
    test("`cargo miri test` (raw-ptr tracking)",
        cargo_miri("test"),
        default_ref, "test.stderr-empty.ref",
        env={'MIRIFLAGS': "-Zmiri-track-raw-pointers"},
    )
    test("`cargo miri test` (with filter)",
        cargo_miri("test") + ["--", "--format=pretty", "le1"],
        filter_ref, "test.stderr-empty.ref",
    )
    test("`cargo miri test` (test target)",
        cargo_miri("test") + ["--test", "test", "--", "--format=pretty"],
        "test.test-target.stdout.ref", "test.stderr-empty.ref",
    )
    test("`cargo miri test` (bin target)",
        cargo_miri("test") + ["--bin", "cargo-miri-test", "--", "--format=pretty"],
        "test.bin-target.stdout.ref", "test.stderr-empty.ref",
    )
    test("`cargo miri test` (subcrate, no isolation)",
        cargo_miri("test") + ["-p", "subcrate"],
        "test.subcrate.stdout.ref", "test.stderr-proc-macro.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.environ["RUST_TEST_NOCAPTURE"] = "0" # this affects test output, so make sure it is not set

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
