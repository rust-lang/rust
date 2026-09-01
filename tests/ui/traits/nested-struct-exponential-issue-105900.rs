//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/105900>.
// Used to take exponentially long

use std::marker::PhantomData;

pub trait Wrappable<I>: Sized {
    type Type;
    fn wrap(self) -> Wrapper<Self, Self::Type> {
        Wrapper(self, PhantomData)
    }
}

pub struct Wrapper<A, E>(A, PhantomData<E>);
impl<I, E, A: Wrappable<I, Type = E>> Wrappable<I> for Wrapper<A, E> {
    type Type = E;
}

impl Wrappable<u8> for u8 {
    type Type = u8;
}

pub fn function() {
    0u8
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap()
        .wrap();
}
fn main() {}
