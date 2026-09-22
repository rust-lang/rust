#![feature(f16b)]

extern crate core;

use core::num::f16b;

// `f16b` is a nominal core type, not a floating-point literal type.
fn main() {
    let value = f16b::from_bits(0x3f80);
    let _: f16b = value;

    let _ = 0.0f16b;
    //~^ ERROR invalid suffix `f16b` for float literal
}
