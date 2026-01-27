//! Regression test for <https://github.com/rust-lang/rust/issues/136379>

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

pub struct S();

impl S {
    pub fn f() -> [u8; S] {
        //~^ ERROR the constant `S` is not of type `usize`
        []
        //~^ ERROR mismatched types [E0308]
    }
}

pub fn main() {}
