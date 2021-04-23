#![feature(member_constraints)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub trait A {
    type B;
    fn f(&self) -> Self::B;
}
impl<'a, 'b> A for () {
    //~^ ERROR the lifetime parameter `'a` is not constrained
    //~| ERROR the lifetime parameter `'b` is not constrained
    type B = impl core::fmt::Debug;

    fn f(&self) -> Self::B {}
}

fn main() {}
