//@ no-prefer-dynamic
//@ aux-build: impl1.rs
//@ aux-build: impl2.rs
//@ aux-build: impl3.rs
//@ ignore-backends: gcc
// Tests the error message when there are multiple implementations of an EII in many crates.
#![feature(eii)]

// has a span but in the other crate
//~? ERROR multiple implementations of `#[eii1]`

extern crate impl1;
extern crate impl2;
extern crate impl3;

fn main() {}
