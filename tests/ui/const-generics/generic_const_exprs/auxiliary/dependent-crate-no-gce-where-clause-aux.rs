#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub fn f<const N: usize>() -> [u8; N + 1]
where
    [(); N + 1]:,
{
    [0; N + 1]
}

pub struct W<const N: usize>
where
    [(); N + 1]:,
{
    pub a: [u8; N + 1],
}

pub fn mk<const N: usize>() -> W<N>
where
    [(); N + 1]:,
{
    W { a: [0; N + 1] }
}

pub trait Tr<const N: usize>
where
    [(); N + 1]:,
{
    fn g(&self) -> [u8; N + 1];
}
