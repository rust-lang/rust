//@ edition:2018
//@ check-pass
//@ run-rustfix
//@ rustfix-only-machine-applicable

#![no_std]
#![crate_type = "lib"]

extern crate std;

pub fn f() {
    std::panic!(123);
    //~^ WARN panic message is not a string literal
}
