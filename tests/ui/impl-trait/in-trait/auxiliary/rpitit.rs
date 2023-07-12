#![feature(return_position_impl_trait_in_trait)]

use std::ops::Deref;

pub trait Foo {
    fn bar(self) -> impl Deref<Target = impl Sized>;
}

pub struct Foreign;
impl Foo for Foreign {
    fn bar(self) -> &'static () {
        &()
    }
}
