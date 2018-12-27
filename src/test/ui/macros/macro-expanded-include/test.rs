// ignore-emscripten no asm! support
// compile-pass
#![feature(asm)]
#![allow(unused)]

#[macro_use]
mod foo;

m!();
fn f() { n!(); }


fn main() {}
