//@ aux-build:implied-predicates.rs
//@ check-pass

extern crate implied_predicates;
use implied_predicates::Bar;

fn bar<B: Bar>() {}

fn main() {}
