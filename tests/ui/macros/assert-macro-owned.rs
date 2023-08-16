// run-fail
//@error-in-other-file:panicked at 'test-assert-owned'
//@ignore-target-emscripten no processes

#![allow(non_fmt_panics)]

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
