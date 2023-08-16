// run-fail
//@error-in-other-file:panicked at 'test-fail-fmt 42 rust'
//@ignore-target-emscripten no processes

fn main() {
    panic!("test-fail-fmt {} {}", 42, "rust");
}
