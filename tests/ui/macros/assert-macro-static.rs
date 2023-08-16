// run-fail
//@error-in-other-file:panicked at 'test-assert-static'
//@ignore-target-emscripten no processes

fn main() {
    assert!(false, "test-assert-static");
}
