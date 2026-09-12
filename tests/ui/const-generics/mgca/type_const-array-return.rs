//@ check-pass
// This test should compile without an ICE.
#![expect(incomplete_features)]
#![feature(min_generic_const_args, macroless_generic_const_args)]

pub struct A;

pub trait Array {
    #[rustc_always_gca]
    const LEN: usize;
    fn arr() -> [u8; Self::LEN];
}

impl Array for A {
    const LEN: usize = core::direct_const_arg!(4);

    #[allow(unused_braces)]
    fn arr() -> [u8; const { Self::LEN }] {
        return [0u8; const { Self::LEN }];
    }
}

fn main() {
    let _ = A::arr();
}
