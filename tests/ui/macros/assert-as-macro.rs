// run-fail
//@error-in-other-file:assertion failed: 1 == 2
//@ignore-target-emscripten no processes

fn main() {
    assert!(1 == 2);
}
