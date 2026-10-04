//@ only-apple
//@ compile-flags: -C link-arg=-lc -C link-arg=-lc
//@ build-fail
#![deny(linker_info)]
//~? ERROR ignoring duplicate libraries
fn main() {}
