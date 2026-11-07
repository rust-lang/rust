//! Checks that `[rev] normalize-*` directives affect the specified revision,
//! and don't affect other revisions.
//!
//! This currently relies on the fact that `normalize-*` directives are
//! applied to run output, not just compiler output. If that ever changes,
//! this test might need to be adjusted.

//@ edition: 2021
//@ revisions: a b
//@ run-pass
//@ check-run-results

//@ normalize-stderr: "output" -> "emitted"
//@[a] normalize-stderr: "first" -> "1st"
//@[b] normalize-stderr: "second" -> "2nd"

fn main() {
    eprintln!("first output line");
    eprintln!("second output line");
}
