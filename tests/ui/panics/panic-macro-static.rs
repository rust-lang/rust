// run-fail
// error-pattern:panicked at 'test-fail-static'
// ignore-emscripten no processes

fn main() {
    panic!("test-fail-static");
}
