//@ revisions: cfail
#![feature(generic_const_exprs, adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct NonZeroUsize(usize);

impl NonZeroUsize {
    const fn get(self) -> usize {
        self.0
    }
}

// regression test for #77650
struct C<T, const N: NonZeroUsize>([T; N.get()])
where
    [T; N.get()]: Sized;
impl<'a, const N: NonZeroUsize, A, B: PartialEq<A>> PartialEq<&'a [A]> for C<B, N>
where
    [B; N.get()]: Sized,
{
    fn eq(&self, other: &&'a [A]) -> bool {
        self.0 == other
        //~^ error: can't compare
    }
}

fn main() {}
