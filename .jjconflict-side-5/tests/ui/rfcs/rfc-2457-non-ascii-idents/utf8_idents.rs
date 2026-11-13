//! Check that non-ascii-idents are allowed.

//@ check-pass
//
#![allow(mixed_script_confusables, non_camel_case_types)]

fn foo<'β, γ>() {}

struct X {
    δ: usize,
}

pub fn main() {
    let α = 0.00001f64;
}
