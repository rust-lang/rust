//@ known-bug: #136379
#![feature(min_generic_const_args)]
pub struct S();

impl S {
    pub fn f() -> [u8; S] {
        []
    }
}

pub fn main() {}
