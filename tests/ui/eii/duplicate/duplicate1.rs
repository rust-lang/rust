//@ no-prefer-dynamic
//@ aux-build: impl1.rs
//@ aux-build: impl2.rs
#![feature(eii)]

// has a span but in the other crate
//~? ERROR multiple implementations of `#[eii1]`

extern crate impl1;
extern crate impl2;

fn main() {}
