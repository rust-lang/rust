// Check rustdoc's test JUnit (XML) output against snapshots.

//@ ignore-cross-compile (running doctests)
//@ ignore-stage1 (rustdoc depends on a fix in libtest)
//@ needs-unwind (test file contains `should_panic` test)

use std::path::Path;

use run_make_support::{cwd, diff, python_command, rustc, rustdoc};

fn main() {
    let rlib = cwd().join("libdoctest.rlib");
    rustc().input("doctest.rs").crate_type("rlib").output(&rlib).run();

    run_doctests(&rlib, "2021", "doctest-2021.xml");
    run_doctests(&rlib, "2024", "doctest-2024.xml");
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

    // FIXME: merged output of compile_fail tests is broken
    if edition != "2024" {
        python_command().arg("validate_junit.py").stdin_buf(rustdoc_stdout).run();
    }

    diff()
        .expected_file(expected_xml)
        .actual_text("output", rustdoc_stdout)
        .normalize(r#"\b(time|total_time|compilation_time)="[0-9.]+""#, r#"$1="$$TIME""#)
        .run();
}
