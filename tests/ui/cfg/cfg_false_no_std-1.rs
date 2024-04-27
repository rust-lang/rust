// No error, panic handler is supplied by libstd linked though the empty library.

//@ check-pass
//@ aux-build: cfg_false_lib_no_std_after.rs

#![no_std]

extern crate cfg_false_lib_no_std_after as _;

fn main() {}
