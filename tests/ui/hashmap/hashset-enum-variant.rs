//! Checks that `HashSet` initialization with enum variants correctly includes only
//! specified variants, preventing platform-specific bugs
//! where all enum variants were mistakenly included
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/42918>

//@ run-pass
//@ compile-flags: -O

#![allow(dead_code)]

use std::collections::HashSet;

#[derive(PartialEq, Debug, Hash, Eq, Clone, PartialOrd, Ord)]
enum MyEnum {
    E0,
    E1,
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
}

fn main() {
    use MyEnum::*;
    let s: HashSet<_> = [E4, E1].iter().cloned().collect();
    let mut v: Vec<_> = s.into_iter().collect();
    v.sort();

    assert_eq!([E1, E4], &v[..]);
}
