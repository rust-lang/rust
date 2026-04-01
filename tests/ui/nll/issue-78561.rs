//@ check-pass
#![feature(impl_trait_in_assoc_type)]

pub trait Trait {
    type A;

    fn f() -> Self::A;
}

pub trait Tr2<'a, 'b> {}

pub struct A<T>(T);
pub trait Tr {
    type B;
}

impl<'a, 'b, T: Tr<B = dyn Tr2<'a, 'b>>> Trait for A<T> {
    type A = impl core::fmt::Debug;

    fn f() -> Self::A {}
}

fn main() {}
