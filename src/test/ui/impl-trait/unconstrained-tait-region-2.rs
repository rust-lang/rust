// check-pass
// edition:2021

#![feature(type_alias_impl_trait)]

use std::future::Future;

pub trait Ctx {}

pub trait MyTrait {
    type AssocT<'m, C>: Future<Output = ()> + 'm
    where
        Self: 'm,
        C: Ctx + 'm;
    fn run<'d, C: Ctx + 'd>(&mut self, c: C) -> Self::AssocT<'_, C>;
}

pub struct MyType;

impl MyTrait for MyType {
    type AssocT<'m, C> = impl Future<Output = ()> + 'm where Self: 'm, C: Ctx + 'm;
    fn run<'d, C: Ctx + 'd>(&mut self, c: C) -> Self::AssocT<'_, C> {
        async move {}
    }
}

fn main() {
    let t = MyType;
}
