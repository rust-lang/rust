// compile-flags: --crate-type=lib
// check-pass

#![feature(return_position_impl_trait_in_trait)]
#![deny(refining_impl_trait)]

pub trait Tr<T> {
    fn foo() -> impl for<'a> Tr<&'a Self>;
}

impl<T> Tr<T> for () {
    fn foo() -> impl for<'a> Tr<&'a Self> {}
}
