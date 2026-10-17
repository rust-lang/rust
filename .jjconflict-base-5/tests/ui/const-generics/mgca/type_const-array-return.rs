//@ check-pass
// This test should compile without an ICE.
#![expect(incomplete_features)]
#![feature(gca_min_const_items, gca_macroless_args)]

use std::gca;

pub struct A;

pub trait Array {
    #[rustc_always_gca]
    const LEN: usize;
    fn arr() -> [u8; Self::LEN];
}

impl Array for A {
    const LEN: usize = gca!(4);

    #[allow(unused_braces)]
    fn arr() -> [u8; const { Self::LEN }] {
        return [0u8; const { Self::LEN }];
    }
}

fn main() {
    let _ = A::arr();
}
