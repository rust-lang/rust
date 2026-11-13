//@ edition:2018
//@ aux-build:dummy-crate.rs
//@ run-rustfix
//@ check-pass

#![warn(rust_2018_idioms)]

extern crate dummy_crate as foo; //~ WARNING unused extern crate
extern crate core; //~ WARNING unused extern crate

mod another {
    extern crate dummy_crate as foo; //~ WARNING unused extern crate
    extern crate core; //~ WARNING unused extern crate
}

fn main() {}
