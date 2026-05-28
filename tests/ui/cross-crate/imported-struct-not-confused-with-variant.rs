//@ run-pass
//@ aux-build:imported-struct-not-confused-with-variant.rs

//! Regression test for https://github.com/rust-lang/rust/issues/19293
//! This test ensures that cross crate variants are properly namespaced under their enum or struct.

extern crate imported_struct_not_confused_with_variant;
use imported_struct_not_confused_with_variant::{Foo, MyEnum};

fn main() {
    MyEnum::Foo(Foo(5));
}
