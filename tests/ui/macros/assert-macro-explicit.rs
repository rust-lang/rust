// run-fail
//@error-in-other-file:panicked at 'assertion failed: false'
//@ignore-target-emscripten no processes

fn main() {
    assert!(false);
}
