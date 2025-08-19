//! This test verifies that tail call optimization does not lead to argument slot leaks.
//!
//! Regression test for: <https://github.com/rust-lang/rust/issues/160>

//@ run-pass

fn inner(dummy: String, b: bool) {
    if b {
        return inner(dummy, false);
    }
}

pub fn main() {
    inner("hi".to_string(), true);
}
