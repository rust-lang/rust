//@ run-pass
//@ aux-build:issue-9968.rs


extern crate issue_9968 as lib;

use lib::{Trait, Struct};

pub fn main()
{
    Struct::init().test();
}

// https://github.com/rust-lang/rust/issues/9968
