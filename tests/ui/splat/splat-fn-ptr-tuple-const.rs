//! Test using `#[splat]` on tuple arguments of generic function constants.

#![allow(incomplete_features)]
#![feature(splat, tuple_trait)]

use std::marker::Tuple;

fn f<Args: Tuple>(#[splat] args: Args) {}

fn main() {
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    const F2: fn(#[splat] (u8, u32)) = f::<(u8, u32)>;
    const R2: () = F2(1, 2); //~ ERROR function pointer calls are not allowed in constants

    #[rustfmt::skip]
    const F1: fn(#[splat] ((u8, u32),)) = f::<((u8, u32),)>;
    const R1: () = F1((1, 2)); //~ ERROR function pointer calls are not allowed in constants
}
