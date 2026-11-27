#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct Foo<const N: usize>;

pub fn mk<const N: usize>(_: Foo<N>) -> [u8; N + 1]
where
    [(); N + 1]:,
{
    [0; N + 1]
}
