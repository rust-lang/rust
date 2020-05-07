// run-fail
// error-pattern:panicked at 'assertion failed: false'
// ignore-emscripten no processes

fn main() {
    assert!(false);
}
