// https://github.com/rust-lang/rust/issues/9968
//@ run-pass
//@ aux-build:aux-9968.rs

extern crate aux_9968 as lib;

use lib::{Trait, Struct};

pub fn main()
{
    Struct::init().test();
}
