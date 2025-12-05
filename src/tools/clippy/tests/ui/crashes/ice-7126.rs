//@ check-pass
// This test requires a feature gated const fn and will stop working in the future.

#![feature(const_btree_len)]

use std::collections::BTreeMap;

struct Foo(usize);
impl Foo {
    fn new() -> Self {
        Self(BTreeMap::len(&BTreeMap::<u8, u8>::new()))
    }
}

fn main() {}
