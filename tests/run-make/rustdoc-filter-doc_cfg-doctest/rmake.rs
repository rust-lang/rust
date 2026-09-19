//! Regression test to ensure that `doc(cfg())` has no impact on the filtered-out doctests.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/147033>.

//@ ignore-cross-compile

use run_make_support::rustdoc;

fn check_rustdoc_test_output(edition: &str) {
    let out = rustdoc().input("foo.rs").edition(edition).arg("--test").run().stdout_utf8();

    // There should be two tests run.
    assert!(out.contains("running 2 test"), "Failed with edition {edition}");
    // They should be in `foo.rs`.
    assert!(out.contains("test foo.rs - f (line 3) ... ok"), "Failed with edition {edition}");
    assert!(
        out.contains("test foo.rs - dummy::f2 (line 11) ... ok"),
        "Failed with edition {edition}"
    );
    // We double-check that the test was run (successfully).
    assert!(
        out.contains("test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"),
        "Failed with edition {edition}",
    );
}

fn main() {
    check_rustdoc_test_output("2015");
    check_rustdoc_test_output("2024");
}
