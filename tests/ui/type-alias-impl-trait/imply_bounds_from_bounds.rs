//@ check-pass

#![feature(impl_trait_in_assoc_type, type_alias_impl_trait)]

pub trait Callable {
    type Output;
    fn call() -> Self::Output;
}

pub type OutputHelper = impl Sized;
impl<'a> Callable for &'a () {
    type Output = OutputHelper;
    #[define_opaque(OutputHelper)]
    fn call() -> Self::Output {}
}

fn test<'a>() -> impl Sized {
    <&'a () as Callable>::call()
}

fn want_static<T: 'static>(_: T) {}

fn test2<'a>() {
    want_static(<&'a () as Callable>::call());
}

fn main() {}
