//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// Invalid `CoerceShared` impls must taint type checking before CTFE.

#![feature(const_block_items)]
#![feature(reborrow)]
#![allow(dead_code, unused_variables)]

use std::marker::CoerceShared;

struct MyMut<'a>(&'a u8);
#[derive(Copy, Clone)]
struct MyRef<'a> {
    x: &'a (),
    y: &'a (),
}

impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}
//~^ ERROR implementing `CoerceShared` does not allow multiple lifetimes or fields to be coerced

const {
    let value = 1;
    consume(MyMut(&value));
}

const fn consume(_: MyRef<'_>) {}

fn main() {}
