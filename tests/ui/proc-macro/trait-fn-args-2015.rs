// Unnamed arguments in trait functions can be passed through proc macros on 2015 edition.

//@ check-pass
//@ aux-build:test-macros.rs

#![allow(anonymous_parameters)]

#[macro_use]
extern crate test_macros;

trait Tr {
    #[identity_attr]
    fn method(u8);
}

fn main() {}
