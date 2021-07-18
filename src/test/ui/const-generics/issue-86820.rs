// Regression test for the ICE described in #86820.

#![allow(unused,dead_code)]
use std::ops::BitAnd;

const C: fn() = || is_set();
fn is_set() {
    0xffu8.bit::<0>();
}

trait Bits {
    fn bit<const I : u8>(self) -> bool;
    //~^ NOTE: the const parameter `I` has type `usize`, but the declaration in trait `Bits::bit` has type `u8`
}

impl Bits for u8 {
    fn bit<const I : usize>(self) -> bool {
    //~^ ERROR: method `bit` has an incompatible const parameter type for trait [E0053]
        let i = 1 << I;
        let mask = u8::from(i);
        mask & self == mask
    }
}

fn main() {}
