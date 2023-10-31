// run-fail
// error-pattern:panicked
// error-pattern:test-fail-static
// ignore-emscripten no processes

fn main() {
    panic!("test-fail-static");
}
