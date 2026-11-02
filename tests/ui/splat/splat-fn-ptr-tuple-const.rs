//! Test using `#[rustc_splat]` on tuple arguments of generic function constants.

#![allow(incomplete_features)]
#![feature(splat, tuple_trait)]

use std::marker::Tuple;

fn f<Args: Tuple>(#[rustc_splat] args: Args) {}

// FIXME(rustfmt): the attribute gets deleted by rustfmt
#[rustfmt::skip]
fn main() {
    const F2: fn(#[rustc_splat] (u8, u32)) = f::<(u8, u32)>;
    const R2: () = F2(1, 2); //~ ERROR function pointer calls are not allowed in constants

    const F1: fn(#[rustc_splat] ((u8, u32),)) = f::<((u8, u32),)>;
    const R1: () = F1((1, 2)); //~ ERROR function pointer calls are not allowed in constants
}
