#![allow(unreachable_code)]

// run-fail
// error-pattern:One
// ignore-emscripten no processes

fn main() {
    panic!("One");
    panic!("Two");
}
