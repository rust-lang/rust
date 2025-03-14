#![allow(unreachable_code)]

//@ run-fail
//@ error-pattern:One
//@ needs-subprocess

fn main() {
    panic!("One");
    panic!("Two");
}
