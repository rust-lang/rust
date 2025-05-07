#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub struct FixedBitSet<const N: usize>;

impl<const N: usize> FixedBitSet<N>
where
    [u8; N.div_ceil(8)]: Sized,
{
    pub fn new() -> Self {
        todo!()
    }
}
