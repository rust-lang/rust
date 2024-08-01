//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]

pub const fn foo() {}

#[const_trait]
pub trait Bar {
    fn bar();
}

impl Bar for () {
    fn bar() {}
}
