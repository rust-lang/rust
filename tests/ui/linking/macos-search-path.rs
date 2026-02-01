//@ only-apple
//@ compile-flags: -C link-arg=-Wl,-L/no/such/file/or/directory
//@ build-fail
#![deny(linker_info)]
//~? ERROR search path
fn main() {}
