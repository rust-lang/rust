// No error, panic handler is supplied by libstd linked though the empty library.

//@ check-pass
//@ aux-build: cfg_false_lib.rs

#![no_std]

extern crate cfg_false_lib as _;

fn main() {}
