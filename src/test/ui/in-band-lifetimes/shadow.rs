#![allow(warnings)]
#![feature(in_band_lifetimes)]

struct Foo<T>(T);

impl Foo<&'s u8> {
    fn bar<'s>(&self, x: &'s u8) {} //~ ERROR shadows a lifetime name
    fn baz(x: for<'s> fn(&'s u32)) {} //~ ERROR shadows a lifetime name
}

fn main() {}
