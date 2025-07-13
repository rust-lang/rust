//@ run-pass
//@ aux-build:struct-and-enum-usage-issue-8044.rs

extern crate struct_and_enum_usage_issue_8044 as minimal;
use minimal::{BTree, leaf};

pub fn main() {
    BTree::<isize> { node: leaf(1) };
}

// https://github.com/rust-lang/rust/issues/8044
