// Ensure that `should_panic` doctests only succeed if the test actually panicked.
// Regression test for <https://github.com/rust-lang/rust/issues/143009>.

//@ needs-target-std

use run_make_support::rustdoc;

fn check_output(output: String, edition: &str) {
    let should_contain = &[
        "test test.rs - bad_exit_code (line 1) ... FAILED",
        "test test.rs - did_not_panic (line 6) ... FAILED",
        "test test.rs - did_panic (line 11) ... ok",
        "---- test.rs - bad_exit_code (line 1) stdout ----
Test executable failed (exit status: 1).",
        "---- test.rs - did_not_panic (line 6) stdout ----
Test didn't panic, but it's marked `should_panic`.",
        "test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out;",
    ];
    for text in should_contain {
        assert!(
            output.contains(text),
            "output doesn't contains (edition: {edition}) {:?}\nfull output: {output}",
            text
        );
    }
}

fn main() {
    check_output(rustdoc().input("test.rs").arg("--test").run_fail().stdout_utf8(), "2015");

    // Same check with the merged doctest feature (enabled with the 2024 edition).
    check_output(
        rustdoc().input("test.rs").arg("--test").edition("2024").run_fail().stdout_utf8(),
        "2024",
    );
}
