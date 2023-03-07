// run-fail
// error-pattern:panicked at 'test-assert-static'
// ignore-emscripten no processes

fn main() {
    assert!(false, "test-assert-static");
}
