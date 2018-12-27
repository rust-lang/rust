#![allow(warnings)]
#![feature(in_band_lifetimes)]

fn foo(x: &u32) {
    let y: &'test u32 = x; //~ ERROR use of undeclared lifetime
}

fn foo2(x: &u32) {}
fn bar() {
    let y: fn(&'test u32) = foo2; //~ ERROR use of undeclared lifetime
}

fn main() {}
