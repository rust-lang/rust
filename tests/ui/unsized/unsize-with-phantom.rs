//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(unsize)]

use std::marker::{PhantomData, Unsize};

fn assert<A, B: ?Sized>()
where
    A: Unsize<B>,
{
}

struct Ptr<A: ?Sized>(PhantomData<A>, A);

trait ToPhantom {
    type T;
}
impl<T: ?Sized> ToPhantom for T {
    type T = PhantomData<fn() -> Self>;
}

struct Ptr2<A: ?Sized>(<A as ToPhantom>::T, A);

trait Identity {
    type T;
}

impl<T> Identity for T {
    type T = T;
}

struct Ptr3<A: ?Sized>(<PhantomData<fn() -> A> as Identity>::T, A);

fn main() {
    assert::<Ptr<[u8; 4]>, Ptr<[u8]>>();
    assert::<Ptr2<[u8; 4]>, Ptr2<[u8]>>();
    assert::<Ptr3<[u8; 4]>, Ptr3<[u8]>>();
}
