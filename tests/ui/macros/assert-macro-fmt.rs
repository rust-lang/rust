// run-fail
//@error-in-other-file:panicked at 'test-assert-fmt 42 rust'
//@ignore-target-emscripten no processes

fn main() {
    assert!(false, "test-assert-fmt {} {}", 42, "rust");
}
