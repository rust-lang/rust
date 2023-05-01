// run-fail
// error-pattern:thread 'main' panicked at
// error-pattern:assertion failed: `(14 != 14)`
// error-pattern: left: `14`
// error-pattern:right: `14`
// ignore-emscripten no processes

fn main() {
    assert_ne!(14, 14);
}
