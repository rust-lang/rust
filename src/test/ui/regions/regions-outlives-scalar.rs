// Test that scalar values outlive all regions.
// Rule OutlivesScalar from RFC 1214.

#![feature(rustc_attrs)]
#![allow(dead_code)]

struct Foo<'a> {
    x: &'a i32,
    y: &'static i32
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
