// run-fail
// error-pattern:thread 'main' panicked at
// error-pattern:assertion failed: `(1 + 1 != 2)`
// error-pattern: error: 1 + 1 definitely should not be 2
// error-pattern:  left: `2`
// error-pattern: right: `2`
// ignore-emscripten no processes

fn main() {
    assert_ne!(1 + 1, 2, "1 + 1 definitely should not be 2");
}
