// Test that we DO NOT warn when lifetime name is used multiple
// arguments, or more than once in a single argument.
//
// check-pass

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// OK: used twice
fn c<'a>(x: &'a u32, y: &'a u32) {}

// OK: used twice
fn d<'a>(x: (&'a u32, &'a u32)) {}

fn main() {}
