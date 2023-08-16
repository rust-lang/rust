// run-fail
//@error-in-other-file:panicked at 'test-fail-owned'
//@ignore-target-emscripten no processes

fn main() {
    panic!("test-fail-owned");
}
