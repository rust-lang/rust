// This test requires a feature gated const fn and will stop working in the future.

#![feature(const_btree_new)]

use std::collections::BTreeMap;

struct Foo(BTreeMap<i32, i32>);
impl Foo {
    fn new() -> Self {
        Self(BTreeMap::new())
    }
}

fn main() {}
