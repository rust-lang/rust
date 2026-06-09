#![feature(pattern_types, rustc_attrs)]
#![feature(pattern_type_macro)]
#![allow(incomplete_features)]
//@ check-pass

//! check that pattern types can actually be part of code that compiles successfully.

use std::pat::pattern_type;

type X = std::num::NonZero<u32>;
type Y = pattern_type!(u32 is 1..);
type Z = Option<pattern_type!(u32 is 1..)>;
struct NonZeroU32New(pattern_type!(u32 is 1..));

fn main() {
    let x: Y = unsafe { std::mem::transmute(42_u32) };
    let z: Z = Some(unsafe { std::mem::transmute(42_u32) });
    match z {
        Some(y) => {
            let _: Y = y;
        }
        None => {}
    }
    let x: X = unsafe { std::mem::transmute(x) };
}
