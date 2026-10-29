//! Regression test for <https://github.com/rust-lang/rust/issues/33461>.
//! This used to ICE as coercion to trait object didn't normalize associated
//! type.
//@ run-pass

#![allow(unused_variables)]
use std::marker::PhantomData;

struct TheType<T> {
    t: PhantomData<T>
}

pub trait TheTrait {
    type TheAssociatedType;
}

impl TheTrait for () {
    type TheAssociatedType = ();
}

pub trait Shape<P: TheTrait> {
    fn doit(&self) {
    }
}

impl<P: TheTrait> Shape<P> for TheType<P::TheAssociatedType> {
}

fn main() {
    let ball = TheType { t: PhantomData };
    let handle: &dyn Shape<()> = &ball;
}
