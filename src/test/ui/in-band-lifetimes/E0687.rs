#![allow(warnings)]
#![feature(in_band_lifetimes)]

fn foo(x: fn(&'a u32)) {} //~ ERROR must be explicitly

fn bar(x: &Fn(&'a u32)) {} //~ ERROR must be explicitly

fn baz(x: fn(&'a u32), y: &'a u32) {} //~ ERROR must be explicitly

struct Foo<'a> { x: &'a u32 }

impl Foo<'a> {
    fn bar(&self, x: fn(&'a u32)) {} //~ ERROR must be explicitly
}

fn main() {}
