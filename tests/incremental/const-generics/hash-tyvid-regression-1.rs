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
fn c<T, const N: NonZeroUsize>()
where
    [T; N.get()]: Sized,
{
    use std::convert::TryFrom;
    <[T; N.get()]>::try_from(())
    //~^ error: the trait bound
    //~| error: the trait bound
    //~| error: mismatched types
}

fn main() {}
