// run-fail
// error-pattern:panicked at 'assertion failed: `(left matches right)`
// error-pattern: left: `2`
// error-pattern:right: `3`: 1 + 1 definitely should be 3'
// ignore-emscripten no processes

#![feature(assert_matches)]

fn main() {
    assert_matches!(1 + 1, 3, "1 + 1 definitely should be 3");
}
