// Ensure we don't ICE when lowering contracts on an associated item.

//@ compile-flags: --crate-type=lib
//@ check-pass

#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;

use core::contracts::requires;

struct Foo;

impl Foo {
    #[requires(align > 0 && (align & (align - 1)) == 0)]
    pub fn foo(align: i32) {}
}
