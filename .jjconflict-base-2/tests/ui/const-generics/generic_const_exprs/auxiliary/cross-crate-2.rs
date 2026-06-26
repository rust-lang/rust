#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub struct Foo<const N: usize>;

impl<const N: usize> Foo<N>
where
    [u8; N.div_ceil(8)]: Sized,
{
    pub fn new() -> Self {
        todo!()
    }
}
