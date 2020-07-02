// run-fail
// error-pattern:assertion failed: `(14) == (15)`
// error-pattern:14: `14`
// error-pattern:15: `15`
// ignore-emscripten no processes

fn main() {
    assert_eq!(14, 15);
}
