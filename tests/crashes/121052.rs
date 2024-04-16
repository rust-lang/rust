//@ known-bug: #121052
#![feature(generic_const_exprs, with_negative_coherence)]

use std::ops::Mul;

pub trait Indices<const N: usize> {
    const NUM_ELEMS: usize;
}

impl<I: Indices<N>, J: Indices<N>, const N: usize> Mul for Tensor<I, N>
where
    I: Concat<J>,
    <I as Concat<J>>::Output: Indices<N>,
    [u8; I::NUM_ELEMS]: Sized,
    [u8; J::NUM_ELEMS]: Sized,
    [u8; <I as Concat<J>>::Output::NUM_ELEMS]: Sized,
{
}

pub trait Concat<J> {}

pub struct Tensor<I: Indices<N>, const N: usize> {}

impl<I: Indices<N>, J: Indices<N>, const N: usize> Mul for Tensor<I, N>
where
    I: Concat<J>,
    <I as Concat<J>>::Output: Indices<N>,
    [u8; I::NUM_ELEMS]: Sized,
    [u8; J::NUM_ELEMS]: Sized,
    [u8; <I as Concat<J>>::Output::NUM_ELEMS]: Sized,
{
}
