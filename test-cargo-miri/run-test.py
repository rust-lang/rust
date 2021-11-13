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
    print("Test stdout or stderr did not match reference!")
    print("--- BEGIN test stdout ---")
    print(stdout, end="")
    print("--- END test stdout ---")
    print("--- BEGIN test stderr ---")
    print(stderr, end="")
    print("--- END test stderr ---")
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
        cargo_miri("run") + ["--bin", "cargo-miri-test", "--", "hello world", '"hello world"', r'he\\llo\"world'],
        "run.args.stdout.ref", "run.args.stderr.ref",
    )
    test("`cargo miri r` (subcrate, no isolation)",
        cargo_miri("r") + ["-p", "subcrate"],
        "run.subcrate.stdout.ref", "run.subcrate.stderr.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )
    test("`cargo miri run` (custom target dir)",
        # Attempt to confuse the argument parser.
        cargo_miri("run") + ["--target-dir=custom-run", "--", "--target-dir=target/custom-run"],
        "run.args.stdout.ref", "run.custom-target-dir.stderr.ref",
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
        env={'MIRIFLAGS': "-Zmiri-tag-raw-pointers"},
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
    test("`cargo miri t` (subcrate, no isolation)",
        cargo_miri("t") + ["-p", "subcrate"],
        "test.subcrate.stdout.ref", "test.stderr-proc-macro.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )
    test("`cargo miri test` (subcrate, doctests)",
        cargo_miri("test") + ["-p", "subcrate", "--doc"],
        "test.stdout-empty.ref", "test.stderr-proc-macro-doctest.ref",
    )
    test("`cargo miri test` (custom target dir)",
        cargo_miri("test") + ["--target-dir=custom-test"],
        default_ref, "test.stderr-empty.ref",
    )
    del os.environ["CARGO_TARGET_DIR"] # this overrides `build.target-dir` passed by `--config`, so unset it
    test("`cargo miri test` (config-cli)",
        cargo_miri("test") + ["--config=build.target-dir=\"config-cli\"", "-Zunstable-options"],
        default_ref, "test.stderr-empty.ref",
    )

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.environ["CARGO_TARGET_DIR"] = "target" # this affects the location of the target directory that we need to check
os.environ["RUST_TEST_NOCAPTURE"] = "0" # this affects test output, so make sure it is not set
os.environ["RUST_TEST_THREADS"] = "1" # avoid non-deterministic output due to concurrent test runs

target_str = " for target {}".format(os.environ['MIRI_TEST_TARGET']) if 'MIRI_TEST_TARGET' in os.environ else ""
print(CGREEN + CBOLD + "## Running `cargo miri` tests{}".format(target_str) + CEND)

if not 'MIRI_SYSROOT' in os.environ:
    # Make sure we got a working sysroot.
    # (If the sysroot gets built later when output is compared, that leads to test failures.)
    subprocess.run(cargo_miri("setup"), check=True)
test_cargo_miri_run()
test_cargo_miri_test()
# Ensure we did not create anything outside the expected target dir.
for target_dir in ["target", "custom-run", "custom-test", "config-cli"]:
    if os.listdir(target_dir) != ["miri"]:
        fail(f"`{target_dir}` contains unexpected files")
    # Ensure something exists inside that target dir.
    os.access(os.path.join(target_dir, "miri", "debug", "deps"), os.F_OK)

print("\nTEST SUCCESSFUL!")
sys.exit(0)
