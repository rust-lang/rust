//@run

//@compile-flags:-C panic=abort
// no-prefer-dynamic
//@ignore-target-macos

#![feature(panic_abort)]

extern crate panic_abort;

fn main() {}
