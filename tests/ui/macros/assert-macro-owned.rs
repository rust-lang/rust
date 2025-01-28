//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-assert-owned
//@ needs-subprocess

#![allow(non_fmt_panics)]

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
