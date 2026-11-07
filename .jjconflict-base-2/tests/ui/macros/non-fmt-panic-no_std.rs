//@ edition:2018
//@ check-pass
//@ run-rustfix
//@ rustfix-only-machine-applicable

#![no_std]
#![crate_type = "lib"]

pub fn f(s: &str) {
    panic!(s);
    //~^ WARN panic message is not a string literal
    core::panic!(s);
    //~^ WARN panic message is not a string literal
}
