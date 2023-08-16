// run-fail
//@error-in-other-file:panicked at 'test-fail-static'
//@ignore-target-emscripten no processes

fn main() {
    panic!("test-fail-static");
}
