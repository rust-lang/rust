// No error, panic handler is supplied by libstd linked though the empty library.

//@ check-pass
//@ aux-build: cfg_false_lib.rs
//@ reference: cfg.attr.crate-level-attrs

#![no_std]

extern crate cfg_false_lib as _;

fn main() {}
