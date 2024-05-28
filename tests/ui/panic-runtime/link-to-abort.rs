//@ run-pass

//@ compile-flags:-C panic=abort
//@ no-prefer-dynamic

#![feature(panic_abort)]

extern crate panic_abort;

fn main() {}
