#![feature(pattern_types, rustc_attrs, const_trait_impl, pattern_type_range_trait)]
#![feature(pattern_type_macro)]
#![allow(incomplete_features)]

//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"

use std::pat::pattern_type;

#[rustc_layout(debug)]
type X = std::num::NonZeroU32; //~ ERROR layout_of
#[rustc_layout(debug)]
type Y = pattern_type!(u32 is 1..); //~ ERROR layout_of
#[rustc_layout(debug)]
type Z = Option<pattern_type!(u32 is 1..)>; //~ ERROR layout_of
#[rustc_layout(debug)]
type A = Option<std::num::NonZeroU32>; //~ ERROR layout_of
#[rustc_layout(debug)]
struct NonZeroU32New(pattern_type!(u32 is 1..)); //~ ERROR layout_of

#[rustc_layout(debug)]
type EMPTY = pattern_type!(u32 is 1..1); //~ ERROR unknown layout

#[rustc_layout(debug)]
type WRAP = pattern_type!(u32 is 1..0); //~ ERROR unknown layout
//~^ ERROR: evaluation panicked: exclusive range end at minimum value of type

#[rustc_layout(debug)]
type WRAP2 = pattern_type!(u32 is 5..2); //~ ERROR unknown layout

#[rustc_layout(debug)]
type SIGN = pattern_type!(i8 is -10..=10); //~ ERROR layout_of

#[rustc_layout(debug)]
type MIN = pattern_type!(i8 is -128..=0); //~ ERROR layout_of

#[rustc_layout(debug)]
type SignedWrap = pattern_type!(i8 is 120..=-120); //~ ERROR unknown layout

fn main() {
    let x: pattern_type!(u32 is 1..) = unsafe { std::mem::transmute(42_u32) };
}

//~? ERROR pattern type ranges cannot wrap: 1..=0
//~? ERROR pattern type ranges cannot wrap: 5..=1
//~? ERROR pattern type ranges cannot wrap: 120..=-120
