// run-fail
// error-pattern:custom message
// ignore-emscripten no processes

fn main() {
    assert!(false, "custom message");
}
