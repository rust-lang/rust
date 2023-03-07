// run-fail
// error-pattern:panicked at 'test-fail-owned'
// ignore-emscripten no processes

fn main() {
    panic!("test-fail-owned");
}
