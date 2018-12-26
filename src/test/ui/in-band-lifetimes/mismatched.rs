#![allow(warnings)]
#![feature(in_band_lifetimes)]

fn foo(x: &'a u32, y: &u32) -> &'a u32 { y } //~ ERROR explicit lifetime required

fn foo2(x: &'a u32, y: &'b u32) -> &'a u32 { y } //~ ERROR lifetime mismatch

fn main() {}
