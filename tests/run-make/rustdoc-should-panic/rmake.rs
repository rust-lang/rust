// Ensure that `should_panic` doctests only succeed if the test actually panicked.
// Regression test for <https://github.com/rust-lang/rust/issues/143009>.

//@ ignore-cross-compile

use run_make_support::rustdoc;

fn check_output(edition: &str, panic_abort: bool) {
    let mut rustdoc_cmd = rustdoc();
    rustdoc_cmd.input("test.rs").arg("--test").edition(edition);
    if panic_abort {
        rustdoc_cmd.args(["-C", "panic=abort"]);
    }
    let output = rustdoc_cmd.run_fail().stdout_utf8();
    let should_contain = &[
        "test test.rs - bad_exit_code (line 1) ... FAILED",
        "test test.rs - did_not_panic (line 6) ... FAILED",
        "test test.rs - did_panic (line 11) ... ok",
        "---- test.rs - bad_exit_code (line 1) stdout ----
Test executable failed (exit status: 1).",
        "---- test.rs - did_not_panic (line 6) stdout ----
Test didn't panic, but it's marked `should_panic` (got unexpected return code 1).",
        "test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out;",
    ];
    for text in should_contain {
        assert!(
            output.contains(text),
            "output (edition: {edition}) doesn't contain {:?}\nfull output: {output}",
            text
        );
    }
}

fn main() {
    check_output("2015", false);

    // Same check with the merged doctest feature (enabled with the 2024 edition).
    check_output("2024", false);

    // Checking that `-C panic=abort` is working too.
    check_output("2015", true);
    check_output("2024", true);
}
