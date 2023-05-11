// run-fail
// error-pattern:assertion failed: `(left != right)`
// error-pattern: left: `14`
// error-pattern:right: `14`
// ignore-emscripten no processes

fn main() {
    assert_ne!(14, 14);
}
