#![allow(warnings)]
#![feature(in_band_lifetimes)]

struct Foo {
    x: &'test u32, //~ ERROR undeclared lifetime
}

enum Bar {
    Baz(&'test u32), //~ ERROR undeclared lifetime
}

fn main() {}
