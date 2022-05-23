// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

//[base] check-fail
//[nll] check-pass
// known-bug

// This should pass, but we end up with `A::Iter<'ai>: Sized` for some specific
// `'ai`. We also know that `for<'at> A::Iter<'at>: Sized` from the definition,
// but we prefer param env candidates. We changed this to preference in #92191,
// but this led to unintended consequences (#93262). Suprisingly, this passes
// under NLL. So only a bug in migrate mode.

#![feature(generic_associated_types)]

use std::marker::PhantomData;

pub trait GenAssoc<T> {
    type Iter<'at>;
    fn iter(&self) -> Self::Iter<'_>;
    fn reborrow<'longt: 'shortt, 'shortt>(iter: Self::Iter<'longt>) -> Self::Iter<'shortt>;
}

pub struct Wrapper<'a, T: 'a, A: GenAssoc<T>> {
    a: A::Iter<'a>,
    _p: PhantomData<T>,
}

impl<'ai, T: 'ai, A: GenAssoc<T>> GenAssoc<T> for Wrapper<'ai, T, A>
where
    A::Iter<'ai>: Clone,
{
    type Iter<'b> = ();
    fn iter<'s>(&'s self) -> Self::Iter<'s> {
        let a = A::reborrow::<'ai, 's>(self.a.clone());
    }

    fn reborrow<'long: 'short, 'short>(iter: Self::Iter<'long>) -> Self::Iter<'short> {
        ()
    }
}

fn main() {}
