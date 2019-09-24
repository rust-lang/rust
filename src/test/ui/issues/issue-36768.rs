// run-pass
// compile-flags:--test
#![deny(private_in_public)]

#[test] fn foo() {}
mod foo {}

#[test] fn core() {}
extern crate core;
