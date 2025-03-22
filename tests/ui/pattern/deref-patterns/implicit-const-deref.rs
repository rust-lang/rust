//! Test that we get an error about structural equality rather than a type error when attempting to
//! use const patterns of library pointer types. I.e. test that we don't implicitly peel it.
#![feature(deref_patterns, inline_const_pat)]
#![allow(incomplete_features)]

const EMPTY: Vec<()> = Vec::new();

fn main() {
    match vec![()] {
        EMPTY => {}
        //~^ ERROR: constant of non-structural type `Vec<()>` in a pattern
        const { Vec::new() } => {}
        //~^ ERROR: constant of non-structural type `Vec<()>` in a pattern
        _ => {}
    }
}
