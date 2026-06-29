//! Regression test for <https://github.com/rust-lang/rust/issues/2761>.
//@ run-fail
//@ error-pattern:custom message
//@ needs-subprocess

fn main() {
    assert!(false, "custom message");
}
