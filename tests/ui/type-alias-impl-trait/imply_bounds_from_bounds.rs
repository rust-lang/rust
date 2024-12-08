//@ check-pass

#![feature(impl_trait_in_assoc_type, type_alias_impl_trait)]

mod foo {
    pub trait Callable {
        type Output;
        fn call() -> Self::Output;
    }

    pub type OutputHelper = impl Sized;
    impl<'a> Callable for &'a () {
        type Output = OutputHelper;
        fn call() -> Self::Output {}
    }
}
use foo::*;

fn test<'a>() -> impl Sized {
    <&'a () as Callable>::call()
}

fn want_static<T: 'static>(_: T) {}

fn test2<'a>() {
    want_static(<&'a () as Callable>::call());
}

fn main() {}
