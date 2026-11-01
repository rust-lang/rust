//! Regression test for <https://github.com/rust-lang/rust/issues/43424>.
//! Test generics in attribute paths are rejected.

#![allow(unused)]

macro_rules! m {
    ($attr_path: path) => {
        #[$attr_path]
        fn f() {}
    }
}

m!(inline<u8>); //~ ERROR: unexpected generic arguments in path

fn main() {}
