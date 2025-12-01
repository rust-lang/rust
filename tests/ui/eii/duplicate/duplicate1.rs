//@ no-prefer-dynamic
//@ aux-build: impl1.rs
//@ aux-build: impl2.rs
//@ ignore-backends: gcc
// tests that EIIs error properly, even if the conflicting implementations live in another crate.
#![feature(extern_item_impls)]

// has a span but in the other crate
//~? ERROR multiple implementations of `#[eii1]`

extern crate impl1;
extern crate impl2;

fn main() {}
