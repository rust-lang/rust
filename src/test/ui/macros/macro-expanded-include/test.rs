// ignore-emscripten no llvm_asm! support
// build-pass (FIXME(62277): could be check-pass?)
#![feature(llvm_asm)]
#![allow(unused)]

#[macro_use]
mod foo;

m!();
fn f() { n!(); }


fn main() {}
