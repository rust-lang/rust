#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct S<const N: usize>
where
    [u8; N.div_ceil(8)]: Sized,
{
    _data: [u8; N.div_ceil(8)],
}

impl<const N: usize> S<N>
where
    [u8; N.div_ceil(8)]: Sized,
{
    pub fn new() -> Self {
        S { _data: [0; N.div_ceil(8)] }
    }
}
