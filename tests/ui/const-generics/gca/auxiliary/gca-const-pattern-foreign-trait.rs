//@ aux-build: gca-const-pattern-stable-trait.rs
//@ compile-flags: -Znext-solver
#![feature(gca_min_const_items, gca_const_items)]
#![expect(incomplete_features)]

extern crate gca_const_pattern_stable_trait as t;

use std::gca;

pub struct S;

impl t::Tr for S {
    const C: usize = gca!(3);
}
