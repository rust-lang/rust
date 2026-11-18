//@ revisions: old current
//@[old] edition: 2021
//@[current] edition: 2024
//@ aux-build: stability.rs
//@[current] check-pass

#[macro_use]
extern crate stability as edition_redirect_stability;

#[cfg(old)]
const _: [(); 1] = [(); redirected_macro!()];
//[old]~^ ERROR use of unstable library feature `edition_redirect_old`

#[cfg(current)]
const _: [(); 2] = [(); redirected_macro!()];

fn main() {}
