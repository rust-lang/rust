// run-fail
// error-pattern:panicked at 'test-assert-owned'
// ignore-emscripten no processes

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
