// run-fail
// error-pattern:thread 'main' panicked at
// error-pattern:assertion failed: `(1 + 1 == 3)`
// error-pattern: error: 1 + 1 definitely should be 3
// error-pattern:  left: `2`
// error-pattern: right: `3`
// ignore-emscripten no processes

fn main() {
    assert_eq!(1 + 1, 3, "1 + 1 definitely should be 3");
}
