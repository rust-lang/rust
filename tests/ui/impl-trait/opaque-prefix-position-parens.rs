//! The type printer wraps a multi-bound `impl Trait` in parens after a prefix
//! type constructor, because `&impl A + B` parses as the ambiguous `(&impl A) + B`.
//! A single-bound opaque needs no parens, and neither does one whose only extra
//! bound is a lifetime that the printer does not render.

pub trait Ta {}
pub trait Tb {}
impl Ta for u8 {}
impl Tb for u8 {}

pub fn one() -> impl Ta { 0u8 }
pub fn two() -> impl Ta + Tb { 0u8 }
pub fn lifetime<'a>(_x: &'a u8) -> impl Ta + 'a { 0u8 }

pub fn takes_unit(_: ()) {}

pub fn a() {
    takes_unit(&one());
    //~^ ERROR mismatched types
}

pub fn b() {
    takes_unit(&two());
    //~^ ERROR mismatched types
}

pub fn c(x: &u8) {
    takes_unit(&lifetime(x));
    //~^ ERROR mismatched types
}

fn main() {}
