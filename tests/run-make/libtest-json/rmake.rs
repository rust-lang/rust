// Check libtest's JSON output against snapshots.

//@ ignore-cross-compile
//@ needs-unwind (test file contains #[should_panic] test)

use run_make_support::{cmd, diff, python_command, rustc};

fn main() {
    rustc().arg("--test").input("f.rs").run();

    run_tests(&[], "output-default.json");
    run_tests(&["--show-output"], "output-stdout-success.json");
}

#[track_caller]
fn run_tests(extra_args: &[&str], expected_file: &str) {
    let cmd_out = cmd("./f")
        .env("RUST_BACKTRACE", "0")
        .args(&["-Zunstable-options", "--test-threads=1", "--format=json"])
        .args(extra_args)
        .run_fail();
    let test_stdout = &cmd_out.stdout_utf8();

    python_command().arg("validate_json.py").stdin(test_stdout).run();

    diff()
        .expected_file(expected_file)
        .actual_text("stdout", test_stdout)
        .normalize(r#"(?<prefix>"exec_time": )[0-9.]+"#, r#"${prefix}"$$EXEC_TIME""#)
        .run();
}
