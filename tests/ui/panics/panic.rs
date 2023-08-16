// run-fail
//@error-in-other-file:1 == 2
//@ignore-target-emscripten no processes

fn main() {
    assert!(1 == 2);
}
