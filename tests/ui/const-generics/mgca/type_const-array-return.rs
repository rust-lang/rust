//@ check-pass
// This test should compile without an ICE.
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

pub struct A;

pub trait Array {
    #[type_const]
    const LEN: usize;
    fn arr() -> [u8; Self::LEN];
}

impl Array for A {
    #[type_const]
    const LEN: usize = 4;

    #[allow(unused_braces)]
    fn arr() -> [u8; const { Self::LEN }] {
        return [0u8; const { Self::LEN }];
    }
}

fn main() {
    let _ = A::arr();
}
