// ignore-test

#![allow(incomplete_features)]
#![feature(const_mut_refs)]
#![feature(inline_const_pat)]

use std::marker::PhantomData;

#[derive(PartialEq, Eq)]
pub struct InvariantRef<'a, T: ?Sized>(&'a T, PhantomData<&'a mut &'a T>);

impl<'a, T: ?Sized> InvariantRef<'a, T> {
    pub const fn new(r: &'a T) -> Self {
        InvariantRef(r, PhantomData)
    }
}

impl<'a> InvariantRef<'a, ()> {
    pub const NEW: Self = InvariantRef::new(&());
}

fn match_invariant_ref<'a>() {
    let y = ();
    match InvariantRef::new(&y) {
    //~^ ERROR `y` does not live long enough [E0597]
        // FIXME(nbdd0121): This should give the same error as `InvariantRef::<'a>::NEW` (without
        // const block)
        const { InvariantRef::<'a>::NEW } => (),
    }
}

fn main() {
    match_invariant_ref();
}
