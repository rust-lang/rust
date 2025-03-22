//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(const_trait_impl)]

// This test can fail because of the implicit const bounds/supertraits.

pub struct Bar;

#[const_trait]
pub trait Foo {}

impl const Foo for Bar {}

pub const fn bar<T: ~const Foo>() {}
