// Check rustdoc's test JUnit (XML) output against snapshots.

//@ ignore-cross-compile (running doctests)
//@ needs-unwind (test file contains `should_panic` test)

use std::path::Path;

use run_make_support::{cwd, diff, python_command, rustc, rustdoc};

fn main() {
    let rlib = cwd().join("libdoctest.rlib");
    rustc().input("doctest.rs").crate_type("rlib").output(&rlib).run();

    run_doctests(&rlib, "2021", "doctest-2021.xml");
    run_doctests_fail(&rlib, "2024");
}

#[track_caller]
fn run_doctests(rlib: &Path, edition: &str, expected_xml: &str) {
    let rustdoc_out = rustdoc()
        .input("doctest.rs")
        .args(&[
            "--test",
            "--test-args=-Zunstable-options",
            "--test-args=--test-threads=1",
            "--test-args=--format=junit",
        ])
        .edition(edition)
        .env("RUST_BACKTRACE", "0")
        .extern_("doctest", rlib.display().to_string())
        .run();
    let rustdoc_stdout = &rustdoc_out.stdout_utf8();

    python_command().arg("validate_junit.py").stdin_buf(rustdoc_stdout).run();

    diff()
        .expected_file(expected_xml)
        .actual_text("output", rustdoc_stdout)
        .normalize(r#"\btime="[0-9.]+""#, r#"time="$$TIME""#)
        .run();
}

// FIXME: gone in the next patch
#[track_caller]
fn run_doctests_fail(rlib: &Path, edition: &str) {
    let rustdoc_out = rustdoc()
        .input("doctest.rs")
        .args(&[
            "--test",
            "--test-args=-Zunstable-options",
            "--test-args=--test-threads=1",
            "--test-args=--format=junit",
        ])
        .edition(edition)
        .env("RUST_BACKTRACE", "0")
        .extern_("doctest", rlib.display().to_string())
        .run_fail();
    let rustdoc_stderr = &rustdoc_out.stderr_utf8();

    diff()
        .expected_text(
            "expected",
            r#"
thread 'main' ($TID) panicked at library/test/src/formatters/junit.rs:22:9:
assertion failed: !s.contains('\n')
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
"#,
        )
        .actual_text("actual", rustdoc_stderr)
        .normalize(r#"thread 'main' \([0-9]+\)"#, r#"thread 'main' ($$TID)"#)
        .run();
}
