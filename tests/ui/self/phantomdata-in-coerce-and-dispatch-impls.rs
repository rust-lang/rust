//@ check-pass

#![feature(coerce_unsized, dispatch_from_dyn, unsize, phantom_variance_markers)]

use std::marker::*;
use std::ops::{CoerceUnsized, DispatchFromDyn};

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

struct W<T: 'static> {
    t: &'static T,
    f: <PhantomData<T> as Mirror>::Assoc,
    cov: PhantomCovariant<T>,
    contra: PhantomContravariant<T>,
    inv: PhantomInvariant<T>,
    cov_lt: PhantomCovariantLifetime<'static>,
    contra_lt: PhantomContravariantLifetime<'static>,
    inv_lt: PhantomInvariantLifetime<'static>,
}

impl<T, U> CoerceUnsized<W<U>> for W<T> where T: Unsize<U> {}

impl<T, U> DispatchFromDyn<W<U>> for W<T> where T: Unsize<U> {}

fn main() {}
