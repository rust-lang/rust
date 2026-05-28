// Benchmarks, when ran as tests, would cause strange indentations
// to appear in the output. This was because padding formatting was
// applied before the conversion from bench to test, and not afterwards.
// Now that this bug has been fixed in #118548, this test checks that it
// does not make a resurgence by comparing the output of --bench with an
// example stdout file.
// See https://github.com/rust-lang/rust/issues/104092

//@ ignore-cross-compile
// Reason: the compiled code is ran
//@ needs-unwind
// Reason: #[bench] requires -Z panic-abort-tests

use run_make_support::{diff, run_with_args, rustc};

fn main() {
    rustc().arg("--test").input("tests.rs").run();
    let out = run_with_args("tests", &["--test-threads=1"]).stdout_utf8();
    diff()
        .expected_file("test.stdout")
        .actual_text("actual-test-stdout", out)
        .normalize(
            // Replace all instances of (arbitrary numbers)
            // [1.2345 ns/iter (+/- 0.1234)]
            // with
            // [?? ns/iter (+/- ??)]
            r#"(\d+(?:[.,]\d+)*)\s*ns/iter\s*\(\+/-\s*(\d+(?:[.,]\d+)*)\)"#,
            "?? ns/iter (+/- ??)",
        )
        // Replace all instances of (arbitrary numbers)
        // finished in 8.0000 s
        // with
        // finished in ??
        .normalize(r#"finished\s+in\s+(\d+(?:\.\d+)*)"#, "finished in ??")
        .run();
    let out = run_with_args("tests", &["--test-threads=1", "--bench"]).stdout_utf8();
    diff()
        .expected_file("bench.stdout")
        .actual_text("actual-bench-stdout", out)
        .normalize(
            r#"(\d+(?:[.,]\d+)*)\s*ns/iter\s*\(\+/-\s*(\d+(?:[.,]\d+)*)\)"#,
            "?? ns/iter (+/- ??)",
        )
        .normalize(r#"finished\s+in\s+(\d+(?:\.\d+)*)"#, "finished in ??")
        .run();
}
