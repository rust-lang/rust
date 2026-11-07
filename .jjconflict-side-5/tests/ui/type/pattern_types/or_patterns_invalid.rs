//! Demonstrate some use cases of or patterns

#![feature(
    pattern_type_macro,
    pattern_types,
    rustc_attrs,
    const_trait_impl,
    pattern_type_range_trait
)]

use std::pat::pattern_type;

fn main() {
    //~? ERROR: only non-overlapping pattern type ranges are allowed at present
    let not_adjacent: pattern_type!(i8 is -127..0 | 1..) = unsafe { std::mem::transmute(0) };
    //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types

    //~? ERROR: one pattern needs to end at `i8::MAX`, but was 29 instead
    let not_wrapping: pattern_type!(i8 is 10..20 | 20..30) = unsafe { std::mem::transmute(0) };
    //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types

    //~? ERROR: only signed integer base types are allowed for or-pattern pattern types
    let not_signed: pattern_type!(u8 is 10.. | 0..5) = unsafe { std::mem::transmute(0) };
    //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types

    //~? ERROR: allowed are two range patterns that are directly connected
    let not_simple_enough_for_mvp: pattern_type!(i8 is ..0 | 1..10 | 10..) =
        unsafe { std::mem::transmute(0) };
        //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
}
