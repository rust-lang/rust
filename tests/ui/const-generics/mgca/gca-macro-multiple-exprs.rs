#![feature(gca_min_const_items)]
use std::gca;
struct S<const N: usize>;
fn foo<const N: usize, const M: usize>(_: S<gca!(N, M)>) {}
//~^ ERROR expected one of
//~| ERROR type provided when a constant was expected
fn main() {}
