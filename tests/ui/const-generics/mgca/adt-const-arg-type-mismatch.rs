//! Tests that if the types differ, only the error "the constant ... is not of type ..."
//! is displayed instead of "mismatched types"
//! whereas if the types match but the values differ,
//! the "mismatched types" error is displayed.
//!
//! issue: https://github.com/rust-lang/rust/issues/162851

#![feature(min_adt_const_params, gca_min_const_items, gca_adts)]
use std::gca;

struct S<const A: [u16; 1]>;

fn main() {
    let _: S<gca!([1_u16])> = S::<gca!([1_i32])>;
    //~^ ERROR: the constant `1` is not of type `u16`

    let _: S<gca!([1_u16])> = S::<gca!([2_u16])>;
    //~^ ERROR: mismatched types
}
