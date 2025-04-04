#!/usr/bin/env python3
'''
Test whether cargo-miri works properly.
Assumes the `MIRI_SYSROOT` env var to be set appropriately,
and the working directory to contain the cargo-miri-test project.
'''

import difflib
import os
import re
import subprocess
import sys
import argparse

CGREEN  = '\33[32m'
CBOLD   = '\33[1m'
CEND    = '\33[0m'

CARGO_EXTRA_FLAGS = os.environ.get("CARGO_EXTRA_FLAGS", "").split()

def fail(msg):
    print("\nTEST FAIL: {}".format(msg))
    sys.exit(1)

def cargo_miri(cmd, quiet = True, targets = None):
    args = ["cargo", "miri", cmd] + CARGO_EXTRA_FLAGS
    if quiet:
        args += ["-q"]

    if targets is not None:
        for target in targets:
            args.extend(("--target", target))
    elif ARGS.target is not None:
        args += ["--target", ARGS.target]

    return args

def normalize_stdout(str):
    str = str.replace("src\\", "src/") # normalize paths across platforms
    str = re.sub("finished in \\d+\\.\\d\\ds", "finished in $TIME", str) # the time keeps changing, obviously
    return str

def check_output(actual, path, name):
    if ARGS.bless:
        # Write the output only if bless is set
        open(path, mode='w').write(actual)
        return True
    expected = open(path).read()
    if expected == actual:
        return True
    print(f"{name} output did not match reference in {path}!")
    print(f"--- BEGIN diff {name} ---")
    for text in difflib.unified_diff(expected.split("\n"), actual.split("\n")):
        print(text)
    print(f"--- END diff {name} ---")
    return False

def test(name, cmd, stdout_ref, stderr_ref, stdin=b'', env=None):
    if env is None:
        env = {}
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
    stdout = normalize_stdout(stdout.decode("UTF-8"))
    stderr = stderr.decode("UTF-8")

    stdout_matches = check_output(stdout, stdout_ref, "stdout")
    stderr_matches = check_output(stderr, stderr_ref, "stderr")

    if p.returncode == 0 and stdout_matches and stderr_matches:
        # All good!
        return
    fail("exit code was {}".format(p.returncode))

def test_no_rebuild(name, cmd, env=None):
    if env is None:
        env = {}
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
        fail("rebuild failed")
    # Also check for 'Running' as a sanity check.
    if stderr.count(" Compiling ") > 0 or stderr.count(" Running ") == 0:
        print("--- BEGIN stderr ---")
        print(stderr, end="")
        print("--- END stderr ---")
        fail("Something was being rebuilt when it should not be (or we got no output)")

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
    # This also covers passing arguments without `--`: Cargo will forward unused positional arguments to the program.
    test("`cargo miri run` (with arguments and target)",
        cargo_miri("run") + ["--bin", "cargo-miri-test", "hello world", '"hello world"', r'he\\llo\"world'],
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
    test("`cargo miri run` (test local crate detection)",
         cargo_miri("run") + ["--package=test-local-crate-detection"],
         "run.local_crate.stdout.ref", "run.local_crate.stderr.ref",
    )

def test_cargo_miri_test():
    # rustdoc is not run on foreign targets
    is_foreign = ARGS.target is not None
    default_ref = "test.cross-target.stdout.ref" if is_foreign else "test.default.stdout.ref"
    filter_ref = "test.filter.cross-target.stdout.ref" if is_foreign else "test.filter.stdout.ref"

    test("`cargo miri test`",
        cargo_miri("test"),
        default_ref, "test.empty.ref",
        env={'MIRIFLAGS': "-Zmiri-seed=4242"},
    )
    test("`cargo miri test` (no isolation, no doctests)",
        cargo_miri("test") + ["--bins", "--tests"], # no `--lib`, we disabled that in `Cargo.toml`
        "test.cross-target.stdout.ref", "test.empty.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )
    test("`cargo miri test` (with filter)",
        cargo_miri("test") + ["--", "--format=pretty", "pl"],
        filter_ref, "test.empty.ref",
    )
    test("`cargo miri test` (test target)",
        cargo_miri("test") + ["--test", "test", "--", "--format=pretty"],
        "test.test-target.stdout.ref", "test.empty.ref",
    )
    test("`cargo miri test` (bin target)",
        cargo_miri("test") + ["--bin", "cargo-miri-test", "--", "--format=pretty"],
        "test.bin-target.stdout.ref", "test.empty.ref",
    )
    test("`cargo miri t` (subcrate, no isolation)",
        cargo_miri("t") + ["-p", "subcrate"],
        "test.subcrate.cross-target.stdout.ref" if is_foreign else "test.subcrate.stdout.ref",
        "test.empty.ref",
        env={'MIRIFLAGS': "-Zmiri-disable-isolation"},
    )
    test("`cargo miri test` (proc-macro crate)",
        cargo_miri("test") + ["-p", "proc_macro_crate"],
        "test.empty.ref", "test.proc-macro.stderr.ref",
    )
    test("`cargo miri test` (custom target dir)",
        cargo_miri("test") + ["--target-dir=custom-test"],
        default_ref, "test.empty.ref",
    )
    del os.environ["CARGO_TARGET_DIR"] # this overrides `build.target-dir` passed by `--config`, so unset it
    test("`cargo miri test` (config-cli)",
        cargo_miri("test") + ["--config=build.target-dir=\"config-cli\""],
        default_ref, "test.empty.ref",
    )
    if ARGS.multi_target:
        test_cargo_miri_multi_target()


def test_cargo_miri_multi_target():
    test("`cargo miri test` (multiple targets)",
        cargo_miri("test", targets = ["aarch64-unknown-linux-gnu", "s390x-unknown-linux-gnu"]),
        "test.multiple_targets.stdout.ref", "test.empty.ref",
    )

args_parser = argparse.ArgumentParser(description='`cargo miri` testing')
args_parser.add_argument('--target', help='the target to test')
args_parser.add_argument('--bless', help='bless the reference files', action='store_true')
args_parser.add_argument('--multi-target', help='run tests related to multiple targets', action='store_true')
ARGS = args_parser.parse_args()

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.environ["CARGO_TARGET_DIR"] = "target" # this affects the location of the target directory that we need to check
os.environ["RUST_TEST_NOCAPTURE"] = "0" # this affects test output, so make sure it is not set
os.environ["RUST_TEST_THREADS"] = "1" # avoid non-deterministic output due to concurrent test runs

target_str = " for target {}".format(ARGS.target) if ARGS.target else ""
print(CGREEN + CBOLD + "## Running `cargo miri` tests{}".format(target_str) + CEND)

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
