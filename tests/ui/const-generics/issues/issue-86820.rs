// Regression test for the ICE described in #86820.

#![allow(unused, dead_code)]
use std::ops::BitAnd;

const C: fn() = || is_set();
fn is_set() {
    Bits::bit::<0>(0xffu8);
}

trait Bits {
    fn bit<const I: u8>(self) -> bool;
}

impl Bits for u8 {
    fn bit<const I: usize>(self) -> bool {
        //~^ ERROR: method `bit` has an incompatible generic parameter for trait `Bits` [E0053]
        let i = 1 << I;
        let mask = u8::from(i);
        mask & self == mask
    }
}

fn main() {}
