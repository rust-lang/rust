//! Regression test for <https://github.com/rust-lang/rust/issues/138009>.
#![feature(gca_min_const_items)]

use std::gca;

#[repr(simd)]
//~^ ERROR SIMD types are experimental and possibly buggy
struct T([isize; gca!(N)]);
//~^ ERROR cannot find value `N` in this scope

static X: T = T();
//~^ ERROR this struct takes 1 argument but 0 arguments were supplied

fn main() {}
