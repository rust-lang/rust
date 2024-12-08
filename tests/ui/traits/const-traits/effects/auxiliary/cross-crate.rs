//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub const fn foo() {}

#[const_trait]
pub trait Bar {
    fn bar();
}

impl Bar for () {
    fn bar() {}
}
