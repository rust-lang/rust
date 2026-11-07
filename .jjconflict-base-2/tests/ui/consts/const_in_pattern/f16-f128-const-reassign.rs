//@ check-pass
// issue: rust-lang/rust#122587

#![feature(f128)]
#![feature(f16)]
#![allow(non_upper_case_globals)]

const h: f16 = 0.0f16;
const q: f128 = 0.0f128;

pub fn main() {
    let h = 0.0f16 else { unreachable!() };
    let q = 0.0f128 else { unreachable!() };
}
