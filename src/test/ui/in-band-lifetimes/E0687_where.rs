#![allow(warnings)]
#![feature(in_band_lifetimes)]

fn bar<F>(x: &F) where F: Fn(&'a u32) {} //~ ERROR must be explicitly

fn baz(x: &impl Fn(&'a u32)) {} //~ ERROR must be explicitly

fn main() {}
