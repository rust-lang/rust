// run-fail
// error-pattern:panicked at 'test-assert-owned'
// ignore-emscripten no processes

#![allow(non_fmt_panic)]

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
