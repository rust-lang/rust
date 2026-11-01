// Test that scalar values outlive all regions.
// Rule OutlivesScalar from RFC 1214.

//@ check-pass
#![allow(dead_code)]

struct Foo<'a> {
    x: &'a i32,
    y: &'static i32
}


fn main() { }
