#![feature(f16b)]

extern crate core;

use core::num::f16b;

fn main() {
    let _: f16b = 1.0;
    //~^ ERROR mismatched types

    let x = f16b::from_bits(0x3f80);
    let _ = x + x;
    //~^ ERROR cannot add `f16b` to `f16b`

    let _ = 1u16 as f16b;
    //~^ ERROR non-primitive cast

    let _ = x as f32;
    //~^ ERROR non-primitive cast
}
