#![feature(const_trait_impl, effects)] //~ WARN the feature `effects` is incomplete

pub const fn foo() {}

#[const_trait]
pub trait Bar {
    fn bar();
}

impl Bar for () {
    fn bar() {}
}
