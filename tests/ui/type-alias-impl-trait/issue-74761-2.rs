#![feature(impl_trait_in_assoc_type)]

pub trait A {
    type B;
    fn f(&self) -> Self::B;
}
impl<'a, 'b> A for () {
    //~^ ERROR the lifetime parameter `'a` is not constrained
    //~| ERROR the lifetime parameter `'b` is not constrained
    type B = impl core::fmt::Debug;

    fn f(&self) -> Self::B {}
    //~^ ERROR expected generic lifetime parameter
}

fn main() {}
