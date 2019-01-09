#![allow(unused_variables)]
// aux-build:derive-union.rs

#[macro_use]
extern crate derive_union;

#[repr(C)]
#[derive(UnionTest)]
union Test {
    a: u8,
}

fn main() {
    let t = Test { a: 0 };
}
