//! Regression test for <https://github.com/rust-lang/rust/issues/29857>.
//! This used to ICE during coherence on Error types.
//@ check-pass

use std::marker::PhantomData;

pub trait Foo<P> {}

impl <P, T: Foo<P>> Foo<P> for Option<T> {}

pub struct Qux<T> (PhantomData<*mut T>);

impl<T> Foo<*mut T> for Option<Qux<T>> {}

pub trait Bar {
    type Output: 'static;
}

impl<T: 'static, W: Bar<Output = T>> Foo<*mut T> for W {}

fn main() {}
