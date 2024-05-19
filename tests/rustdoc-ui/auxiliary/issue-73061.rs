//@edition:2018

#![feature(impl_trait_in_assoc_type)]

pub trait Foo {
    type X: std::future::Future<Output = ()>;
    fn x(&self) -> Self::X;
}

pub struct F;

impl Foo for F {
    type X = impl std::future::Future<Output = ()>;
    fn x(&self) -> Self::X {
        async {}
    }
}
