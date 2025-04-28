//! Check where literals can be used to initialize pattern types and where not.

#![feature(pattern_types, const_trait_impl, pattern_type_range_trait)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

fn out_of_range() -> pattern_type!(u32 is 1..) {
    0
    //~^ ERROR mismatched types
}

fn at_range_start() -> pattern_type!(u32 is 1..) {
    1
}

fn in_range() -> pattern_type!(u32 is 1..) {
    2
}

fn negative_lit_on_unsigned_ty() -> pattern_type!(u32 is 1..) {
    -3
    //~^ ERROR: cannot apply unary operator `-` to type `(u32) is 1..`
}

fn negative_lit_in_range() -> pattern_type!(i8 is -5..5) {
    -2
    //~^ ERROR: cannot apply unary operator `-` to type `(i8) is -5..=4`
}

fn positive_lit_in_range_of_signed() -> pattern_type!(i8 is -5..5) {
    2
}

fn negative_lit_at_range_start() -> pattern_type!(i8 is -5..5) {
    -5
    //~^ ERROR mismatched types
}

fn positive_lit_at_range_end() -> pattern_type!(i8 is -5..5) {
    4
}

fn lit_one_beyond_range_end() -> pattern_type!(i8 is -5..5) {
    5
    //~^ ERROR mismatched types
}

fn wrong_lit_kind() -> pattern_type!(u32 is 1..) {
    '3'
    //~^ ERROR mismatched types
}

fn char_lit_in_range() -> pattern_type!(char is 'a'..'z') {
    'b'
    //~^ ERROR mismatched types
}

fn char_lit_out_of_range() -> pattern_type!(char is 'a'..'z') {
    'A'
    //~^ ERROR mismatched types
}

fn lit_at_unsigned_range_inclusive_end() -> pattern_type!(u32 is 0..=1) {
    1
}

fn single_element_range() -> pattern_type!(u32 is 0..=0) {
    0
}

fn lit_oob_single_element_range() -> pattern_type!(u32 is 0..=0) {
    1
    //~^ ERROR mismatched types
}

fn lit_oob_single_element_range_exclusive() -> pattern_type!(u32 is 0..1) {
    1
    //~^ ERROR mismatched types
}

fn single_element_range_exclusive() -> pattern_type!(u32 is 0..1) {
    0
}

fn empty_range_at_base_type_min() -> pattern_type!(u32 is 0..0) {
    //~^ ERROR evaluation of constant value failed
    0
}

fn empty_range_at_base_type_min2() -> pattern_type!(u32 is 0..0) {
    //~^ ERROR evaluation of constant value failed
    1
}

fn empty_range() -> pattern_type!(u32 is 1..1) {
    0
    //~^ ERROR mismatched types
}

fn empty_range2() -> pattern_type!(u32 is 1..1) {
    1
    //~^ ERROR mismatched types
}

fn wraparound_range_at_base_ty_end() -> pattern_type!(u32 is 1..0) {
    //~^ ERROR evaluation of constant value failed
    1
}

fn wraparound_range_at_base_ty_end2() -> pattern_type!(u32 is 1..0) {
    //~^ ERROR evaluation of constant value failed
    0
}

fn wraparound_range_at_base_ty_end3() -> pattern_type!(u32 is 1..0) {
    //~^ ERROR evaluation of constant value failed
    2
}

fn wraparound_range() -> pattern_type!(u32 is 2..1) {
    1
    //~^ ERROR mismatched types
}

fn lit_in_wraparound_range() -> pattern_type!(u32 is 2..1) {
    0
    //~^ ERROR mismatched types
}

fn lit_at_wraparound_range_start() -> pattern_type!(u32 is 2..1) {
    2
    //~^ ERROR mismatched types
}

fn main() {}

//~? ERROR pattern type ranges cannot wrap: 1..=0
//~? ERROR pattern type ranges cannot wrap: 2..=0
