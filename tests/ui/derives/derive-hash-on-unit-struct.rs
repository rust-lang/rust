//! Regression test for <https://github.com/rust-lang/rust/issues/16530>.
//! Tests that unit struct produce same constant hash instead of ICE'ing.

//@ run-pass
#![allow(deprecated)]

use std::hash::{SipHasher, Hasher, Hash};

#[derive(Hash)]
struct Empty;

pub fn main() {
    let mut s1 = SipHasher::new();
    Empty.hash(&mut s1);
    let mut s2 = SipHasher::new();
    Empty.hash(&mut s2);
    assert_eq!(s1.finish(), s2.finish());
}
