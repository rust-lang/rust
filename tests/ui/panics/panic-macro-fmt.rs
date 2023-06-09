// run-fail
// error-pattern:panicked at 'test-fail-fmt 42 rust'
// ignore-emscripten no processes

fn main() {
    panic!("test-fail-fmt {} {}", 42, "rust");
}
