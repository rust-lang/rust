// ICE: assertion failed: !value.has_infer()
// issue: rust-lang/rust#115806
#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]

pub struct NoPin;

impl<TA> Pins<TA> for NoPin {}

pub trait PinA<PER> {
    type const A: &'static () = const { &() };
}

pub trait Pins<USART> {}

impl<USART, T> Pins<USART> for T where T: PinA<USART, A = const { &() }> {}
//~^ ERROR conflicting implementations of trait `Pins<_>` for type `NoPin`

pub fn main() {}
