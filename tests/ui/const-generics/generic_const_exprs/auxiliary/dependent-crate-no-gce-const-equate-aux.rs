#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Tr<const M: usize> {
    fn v(&self) -> usize {
        M
    }
}

pub struct Foo<const N: usize>;

impl<const N: usize> Tr<{ N + 1 }> for Foo<N> {}

pub fn g<const N: usize>(_: [u8; N]) -> [u8; N + 1]
where
    [(); N + 1]:,
{
    [0; N + 1]
}
