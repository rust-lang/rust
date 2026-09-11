// Regression test for https://github.com/rust-lang/rust/issues/154748 and https://github.com/rust-lang/rust/issues/154750

#![feature(min_generic_const_args)]

//@ compile-flags: --emit=mir

const CONST: usize = core::direct_const_arg!(1u32);
//~^ ERROR the constant `1` is not of type `usize`

const S: bool = core::direct_const_arg!(1i32);
//~^ ERROR the constant `1` is not of type `bool`

fn main() {
    CONST;
    S;
}
