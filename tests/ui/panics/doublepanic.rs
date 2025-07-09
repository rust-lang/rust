#![allow(unreachable_code)]

//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    panic!("One");
    panic!("Two");
}
