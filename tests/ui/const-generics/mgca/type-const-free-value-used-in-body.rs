// Regression test for https://github.com/rust-lang/rust/issues/154748 and https://github.com/rust-lang/rust/issues/154750

#![feature(gca_min_const_items)]

//@ compile-flags: --emit=mir

use std::gca;

const CONST: usize = gca!(1u32);
//~^ ERROR the constant `1` is not of type `usize`

const S: bool = gca!(1i32);
//~^ ERROR the constant `1` is not of type `bool`

fn main() {
    CONST;
    S;
}
