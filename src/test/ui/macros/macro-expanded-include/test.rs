// ignore-emscripten no asm! support

#![feature(asm, rustc_attrs)]
#![allow(unused)]

#[macro_use]
mod foo;

m!();
fn f() { n!(); }

#[rustc_error]
fn main() {} //~ ERROR compilation successful
