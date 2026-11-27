// ICE: assertion failed: !value.has_infer()
// issue: rust-lang/rust#115806
#![feature(adt_const_params, gca_min_const_items, unsized_const_params)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]

use std::gca;

pub struct NoPin;

impl<TA> Pins<TA> for NoPin {}

pub trait PinA<PER> {
    #[rustc_always_gca]
    const A: &'static () = gca!(const { &() });
}

pub trait Pins<USART> {}

impl<USART, T> Pins<USART> for T
//~^ ERROR conflicting implementations of trait `Pins<_>` for type `NoPin`
where
    T: PinA<USART, A = { gca!(const { &() }) }>
{
}

pub fn main() {}
