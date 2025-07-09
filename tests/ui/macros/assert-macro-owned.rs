//@ run-fail
//@ check-run-results
//@ needs-subprocess

#![allow(non_fmt_panics)]

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
