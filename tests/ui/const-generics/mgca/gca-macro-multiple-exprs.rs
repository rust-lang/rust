#![feature(min_generic_const_args)]
use std::gca;
struct S<const N: usize>;
fn foo<const N: usize, const M: usize>(_: S<gca!(N, M)>) {}
//~^ ERROR gca! takes 1 argument
fn main() {}
