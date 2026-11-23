//@ compile-flags: -Znext-solver

#![feature(gca_min_const_items, gca_const_items)]

pub const N: usize = std::gca!(1);
