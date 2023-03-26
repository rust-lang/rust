// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty

#![feature(return_position_impl_trait_in_trait)]

pub trait Foo {
    fn bar() -> impl Sized;
}

pub struct Foreign;

impl Foo for Foreign {
    fn bar() {}
}
