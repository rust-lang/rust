// run-fail
// error-pattern:assertion failed: 1 == 2
// ignore-emscripten no processes

fn main() {
    assert!(1 == 2);
}
