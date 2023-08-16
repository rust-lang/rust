#![allow(unreachable_code)]

// run-fail
//@error-in-other-file:One
//@ignore-target-emscripten no processes

fn main() {
    panic!("One");
    panic!("Two");
}
