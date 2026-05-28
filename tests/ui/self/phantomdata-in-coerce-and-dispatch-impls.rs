//@ check-pass

#![feature(coerce_unsized, dispatch_from_dyn, unsize)]

use std::marker::Unsize;
use std::ops::{CoerceUnsized, DispatchFromDyn};
use std::marker::PhantomData;

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

struct W<T: 'static> {
    t: &'static T,
    f: <PhantomData<T> as Mirror>::Assoc,
}

impl<T, U> CoerceUnsized<W<U>> for W<T> where T: Unsize<U> {}

impl<T, U> DispatchFromDyn<W<U>> for W<T> where T: Unsize<U> {}

fn main() {}
