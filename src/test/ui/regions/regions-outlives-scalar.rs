// Test that scalar values outlive all regions.
// Rule OutlivesScalar from RFC 1214.

// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]

struct Foo<'a> {
    x: &'a i32,
    y: &'static i32
}


fn main() { }
