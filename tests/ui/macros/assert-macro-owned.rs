//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:test-assert-owned
//@ ignore-emscripten no processes

#![allow(non_fmt_panics)]

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
