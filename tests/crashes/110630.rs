//@ known-bug: #110630

#![feature(generic_const_exprs)]

use std::ops::Mul;

pub trait Indices<const N: usize> {
    const NUM_ELEMS: usize = I::NUM_ELEMS * N;
}

pub trait Concat<J> {
    type Output;
}

pub struct Tensor<I: Indices<N>, const N: usize>
where
    [u8; I::NUM_ELEMS]: Sized, {}

impl<I: Indices<N>, J: Indices<N>, const N: usize> Mul<Tensor<J, N>> for Tensor<I, N>
where
    I: Concat<T>,
    <I as Concat<J>>::Output: Indices<N>,
    [u8; I::NUM_ELEMS]: Sized,
    [u8; J::NUM_ELEMS]: Sized,
    [u8; <I as Concat<J>>::Output::NUM_ELEMS]: Sized,
{
    type Output = Tensor<<I as Concat<J>>::Output, N>;
}
