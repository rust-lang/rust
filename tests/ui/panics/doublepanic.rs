#![allow(unreachable_code)]

//@ run-fail
//@ check-run-results:One
//@ ignore-emscripten no processes

fn main() {
    panic!("One");
    panic!("Two");
}
