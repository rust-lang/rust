// ignore-emscripten no asm! support
// build-pass (FIXME(62277): could be check-pass?)
#![feature(asm)]
#![allow(unused)]

#[macro_use]
mod foo;

m!();
fn f() { n!(); }


fn main() {}
