//edition:2018

#![feature(min_type_alias_impl_trait)]

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
