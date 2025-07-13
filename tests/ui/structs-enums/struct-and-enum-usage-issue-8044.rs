//@ run-pass
//@ aux-build:issue-8044.rs


extern crate issue_8044 as minimal;
use minimal::{BTree, leaf};

pub fn main() {
    BTree::<isize> { node: leaf(1) };
}

// https://github.com/rust-lang/rust/issues/8044
