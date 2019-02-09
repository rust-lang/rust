// run-pass
#![allow(unused_must_use)]
#![allow(unused_imports)]
// pretty-expanded FIXME(#23616)
#![allow(deprecated)]

use std::hash::{Hash, SipHasher};

// Testing multiple separate deriving attributes.
#[derive(PartialEq)]
#[derive(Clone)]
#[derive(Hash)]
struct Foo {
    bar: usize,
    baz: isize
}

fn hash<T: Hash>(_t: &T) {}

pub fn main() {
    let a = Foo {bar: 4, baz: -3};

    // Check for `PartialEq` impl without testing its correctness.
    a == a;
    // Check for `Clone` impl without testing its correctness.
    a.clone();
    // Check for `Hash` impl without testing its correctness.
    hash(&a);
}
