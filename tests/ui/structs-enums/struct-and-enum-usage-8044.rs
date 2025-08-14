// https://github.com/rust-lang/rust/issues/8044
//@ run-pass
//@ aux-build:aux-8044.rs

extern crate aux_8044 as minimal;
use minimal::{BTree, leaf};

pub fn main() {
    BTree::<isize> { node: leaf(1) };
}
