// ICE unknown alias DefKind: AnonConst
// issue: rust-lang/rust#116710
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct A<const N: u32 = 1, const M: u32 = u8>;
//~^ ERROR expected value, found builtin type `u8`

trait Trait {}
impl<const N: u32> Trait for A<N> {}

impl<const N: u32> Trait for A<N> {}

pub fn main() {}
