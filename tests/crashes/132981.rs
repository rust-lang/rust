//@ known-bug: #132981
//@compile-flags: -Clink-dead-code=true --crate-type lib
//@ only-x86_64
//@ ignore-windows

#![feature(rust_cold_cc)]
pub extern "rust-cold" fn foo(_: [usize; 3]) {}
