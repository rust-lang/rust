//@ run-pass
//@ compile-flags:--test
#![deny(private_interfaces)]

#[test] fn foo() {}
mod foo {}

#[test] fn core() {}
extern crate core;
