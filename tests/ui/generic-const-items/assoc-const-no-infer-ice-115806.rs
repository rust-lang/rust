// ICE: assertion failed: !value.has_infer()
// issue: rust-lang/rust#115806
#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]

pub struct NoPin;

impl<TA> Pins<TA> for NoPin {}

pub trait PinA<PER> {
    #[rustc_always_gca]
    const A: &'static () = core::direct_const_arg!(const { &() });
    //~^ ERROR anonymous constants with lifetimes in their type are not yet supported
}

pub trait Pins<USART> {}

impl<USART, T> Pins<USART> for T
//~^ ERROR conflicting implementations of trait `Pins<_>` for type `NoPin`
where
    T: PinA<USART, A = { core::direct_const_arg!(const { &() }) }>,
    //~^ ERROR anonymous constants with lifetimes in their type are not yet supported
{
}

pub fn main() {}
