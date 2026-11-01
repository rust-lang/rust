// Regression test for https://github.com/rust-lang/rust/issues/152962

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ compile-flags: -Zvalidate-mir

#![feature(gca_min_const_items, gca_macroless_args)]

use std::gca;

pub struct A;

pub trait Array {
    #[rustc_always_gca]
    const LEN: usize;
    fn arr() -> [u8; Self::LEN];
}

impl Array for A {
    const LEN: usize = gca!(0u8);
    //~^ ERROR the constant `0` is not of type `usize`

    fn arr() -> [u8; const { Self::LEN }] {}
    //[current]~^ ERROR the constant `0` is not of type `usize`
}

fn main() {}
