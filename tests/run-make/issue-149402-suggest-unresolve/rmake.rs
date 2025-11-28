//! Check that unstable name-resolution suggestions are omitted on stable.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/149402>.
//!
//@ only-nightly

use run_make_support::{diff, rustc, similar};

fn main() {
    let stable_like = rustc()
        .env("RUSTC_BOOTSTRAP", "-1")
        .edition("2024")
        .input("foo.rs")
        .run_fail()
        .stderr_utf8();

    assert!(!stable_like.contains("CoroutineState::Complete"));
    diff().expected_file("stable.err").actual_text("stable_like", &stable_like).run();

    let nightly = rustc().edition("2024").input("foo.rs").run_fail().stderr_utf8();

    assert!(nightly.contains("CoroutineState::Complete"));
    diff().expected_file("nightly.err").actual_text("nightly", &nightly).run();

    let stderr_diff =
        similar::TextDiff::from_lines(&stable_like, &nightly).unified_diff().to_string();
    diff().expected_file("output.diff").actual_text("diff", stderr_diff).run();
}
