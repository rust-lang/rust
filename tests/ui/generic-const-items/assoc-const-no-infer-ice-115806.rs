// ICE: assertion failed: !value.has_infer()
// issue: rust-lang/rust#115806
#![feature(min_generic_const_args, unsized_const_params)]
#![allow(incomplete_features)]

pub struct NoPin;

impl<TA> Pins<TA> for NoPin {}

pub trait PinA<PER> {
    #[type_const]
    const A: &'static () = const { &() };
}

pub trait Pins<USART> {}

impl<USART, T> Pins<USART> for T where T: PinA<USART, A = const { &() }> {}
//~^ ERROR conflicting implementations of trait `Pins<_>` for type `NoPin`

pub fn main() {}
