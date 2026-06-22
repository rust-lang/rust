// Regression test for https://github.com/rust-lang/rust/issues/154748 and https://github.com/rust-lang/rust/issues/154750

#![feature(min_generic_const_args)]

//@ compile-flags: --emit=mir

type const CONST: usize = 1u32;
//~^ ERROR the constant `1` is not of type `usize`

type const S: bool = 1i32;
//~^ ERROR the constant `1` is not of type `bool`

fn main() {
    CONST;
    S;
}
