extern crate core;

use core::num::f16b;
//~^ ERROR use of unstable library feature `f16b`

fn main() {
    let _ = f16b::from_bits(0);
    //~^ ERROR use of unstable library feature `f16b`
    //~| ERROR use of unstable library feature `f16b`

    let a = 0.0f16b;
    //~^ ERROR  invalid suffix `f16b`

    let _: f16b = 1.0;
    //~^ ERROR use of unstable library feature `f16b`
    //~| ERROR mismatched types

    let x = f16b::from_bits(0x3f80);
    //~^ ERROR use of unstable library feature `f16b`
    //~| ERROR use of unstable library feature `f16b`
    let _ = x + x;
    //~^ ERROR cannot add `f16b` to `f16b`

    let _ = 1u16 as f16b;
    //~^ ERROR use of unstable library feature `f16b`
    //~| ERROR non-primitive cast

    let _ = x as f32;
    //~^ ERROR non-primitive cast
}
