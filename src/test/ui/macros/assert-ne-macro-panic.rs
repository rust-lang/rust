// run-fail
// error-pattern:assertion failed: `(14) != (14)`
// error-pattern:14: `14`
// error-pattern:14: `14`
// ignore-emscripten no processes

fn main() {
    assert_ne!(14, 14);
}
