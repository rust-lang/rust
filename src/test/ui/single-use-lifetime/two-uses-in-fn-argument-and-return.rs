// Test that we DO NOT warn when lifetime name is used in
// both the argument and return.
//
// build-pass (FIXME(62277): could be check-pass?)

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn c<'a>(x: &'a u32) -> &'a u32 { // OK: used twice
    &22
}

fn main() { }
