// aux-build:rlib-crate-test.rs
// ignore-stage1
// ignore-cross-compile gives a different error message

#![feature(plugin)]
#![plugin(rlib_crate_test)]
//~^ ERROR: plugin `rlib_crate_test` only found in rlib format, but must be available in dylib

fn main() {}
