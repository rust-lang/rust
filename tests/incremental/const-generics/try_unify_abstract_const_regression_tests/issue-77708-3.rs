//@ revisions: rpass
#![feature(generic_const_exprs, adt_const_params)]
#![allow(incomplete_features)]

use std::{convert::TryFrom};

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct NonZeroUsize(usize);

impl NonZeroUsize {
    const fn get(self) -> usize {
        self.0
    }
}

struct A<const N: NonZeroUsize>([u8; N.get()])
where
    [u8; N.get()]: Sized;

impl<'a, const N: NonZeroUsize> TryFrom<&'a [u8]> for A<N>
where
    [u8; N.get()]: Sized,
{
    type Error = ();
    fn try_from(slice: &'a [u8]) -> Result<A<N>, ()> {
        let _x = <&[u8; N.get()] as TryFrom<&[u8]>>::try_from(slice);
        unimplemented!();
    }
}

fn main() {}
