// run-fail
//@error-in-other-file:custom message
//@ignore-target-emscripten no processes

fn main() {
    assert!(false, "custom message");
}
