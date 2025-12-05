#![feature(never_type)]
#![feature(transmutability)]

use std::mem::{Assume, TransmuteFrom};

pub fn is_transmutable<Src, Dst>()
where
    Dst: TransmuteFrom<Src, { Assume::SAFETY }>,
{
}

fn main() {
    is_transmutable::<char, u32>();

    // `char`s can be in the following ranges:
    // - [0, 0xD7FF]
    // - [0xE000, 10FFFF]
    //
    // `Char` has variants whose tags are in the top and bottom of each range.
    // It also has generic variants which are out of bounds of these ranges, but
    // are generic on types which may be set to `!` to "disable" them in the
    // transmutability analysis.
    #[repr(u32)]
    enum Char<B, C, D> {
        A = 0,
        B = 0xD7FF,
        OverB(B) = 0xD800,
        UnderC(C) = 0xDFFF,
        C = 0xE000,
        D = 0x10FFFF,
        OverD(D) = 0x110000,
    }

    is_transmutable::<Char<!, !, !>, char>();
    is_transmutable::<Char<(), !, !>, char>();
    //~^ ERROR cannot be safely transmuted into `char`
    is_transmutable::<Char<!, (), !>, char>();
    //~^ ERROR cannot be safely transmuted into `char`
    is_transmutable::<Char<!, !, ()>, char>();
    //~^ ERROR cannot be safely transmuted into `char`
}
