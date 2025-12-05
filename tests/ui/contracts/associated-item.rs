// Ensure we don't ICE when lowering contracts on an associated item.

//@ compile-flags: --crate-type=lib
//@ check-pass

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use

extern crate core;

use core::contracts::requires;

struct Foo;

impl Foo {
    #[requires(align > 0 && (align & (align - 1)) == 0)]
    pub fn foo(align: i32) {}
}
