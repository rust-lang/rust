// Check libtest's JUnit (XML) output against snapshots.

//@ ignore-cross-compile
//@ needs-unwind (test file contains #[should_panic] test)

use run_make_support::{cmd, diff, python_command, rustc};

fn main() {
    rustc().arg("--test").input("f.rs").run();

    run_tests(&[], "output-default.xml");
    run_tests(&["--show-output"], "output-stdout-success.xml");
}

#[track_caller]
fn run_tests(extra_args: &[&str], expected_file: &str) {
    let cmd_out = cmd("./f")
        .env("RUST_BACKTRACE", "0")
        .args(&["-Zunstable-options", "--test-threads=1", "--format=junit"])
        .args(extra_args)
        .run_fail();
    let test_stdout = &cmd_out.stdout_utf8();

    python_command().arg("validate_junit.py").stdin_buf(test_stdout).run();

    diff()
        .expected_file(expected_file)
        .actual_text("stdout", test_stdout)
        .normalize(r#"\btime="[0-9.]+""#, r#"time="$$TIME""#)
        .normalize(r"thread '(?P<name>.*?)' \(\d+\) panicked", "thread '$name' ($$TID) panicked")
        .run();
}
