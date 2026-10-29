#![feature(pattern_types, rustc_attrs)]
#![feature(pattern_type_macro)]
#![allow(incomplete_features)]

//! Some practical niche checks.

use std::pat::pattern_type;

type Z = Option<pattern_type!(u32 is 1..)>;

fn main() {
    let z: Z = Some(unsafe { std::mem::transmute(42_u32) });
    let _: Option<u32> = unsafe { std::mem::transmute(z) }; //~ ERROR: different sizes
}
