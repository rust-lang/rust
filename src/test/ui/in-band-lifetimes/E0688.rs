#![allow(warnings)]
#![feature(in_band_lifetimes)]

fn foo<'a>(x: &'a u32, y: &'b u32) {} //~ ERROR cannot mix

struct Foo<'a> { x: &'a u32 }

impl Foo<'a> {
    fn bar<'b>(x: &'a u32, y: &'b u32, z: &'c u32) {} //~ ERROR cannot mix
}

impl<'b> Foo<'a> { //~ ERROR cannot mix
    fn baz() {}
}

fn main() {}
