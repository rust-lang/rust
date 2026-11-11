//! Regression test for an ICE in const eval when a `CoerceShared` coercion goes through an impl
//! that coherence has already rejected.
//! See <https://github.com/rust-lang/rust/issues/158149>.

#![feature(const_block_items)]
#![feature(reborrow)]
use std::marker::CoerceShared;
struct MyMut<'a>(&'a u8);
struct MyRef<'a> {
    x: &'a (),
    y: &'a (),
}

impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}
//~^ ERROR implementing `CoerceShared` requires source and target structs to use the same field style
const {
    let value = 1;
    foo(MyMut(&value));
}
const fn foo(x: MyRef) {}

fn main() {}
