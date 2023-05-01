// run-fail
// error-pattern:thread 'main' panicked at
// error-pattern:assertion failed: `(14 == 15)`
// error-pattern: left: `14`
// error-pattern:right: `15`
// ignore-emscripten no processes

fn main() {
    assert_eq!(14, 15);
}
