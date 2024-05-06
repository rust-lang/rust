// ICE assertion failed: matches!(self.def_kind(ct.def.did), DefKind :: AnonConst)
// issue: rust-lang/rust#108220
//@ check-pass

#![feature(associated_const_equality)]
#![allow(unused)]

use std::marker::PhantomData;

pub struct NoPin;

pub trait SetAlternate<const A: u8> {}

impl SetAlternate<0> for NoPin {}

pub trait PinA<PER> {
    const A: u8;
}

impl<PER> PinA<PER> for NoPin {
    const A: u8 = 0;
}

pub trait Pins<USART> {}

impl<USART, T, const TA: u8> Pins<USART> for T where
    T: PinA<USART, A = { TA }> + SetAlternate<TA>
{
}

struct Serial<USART>(PhantomData<USART>);

impl<USART> Serial<USART> where NoPin: Pins<USART> {}

fn main() {}
