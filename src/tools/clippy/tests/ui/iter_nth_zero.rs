// run-rustfix

#![warn(clippy::iter_nth_zero)]
use std::collections::HashSet;

struct Foo {}

impl Foo {
    fn nth(&self, index: usize) -> usize {
        index + 1
    }
}

fn main() {
    let f = Foo {};
    f.nth(0); // lint does not apply here

    let mut s = HashSet::new();
    s.insert(1);
    let _x = s.iter().nth(0);

    let mut s2 = HashSet::new();
    s2.insert(2);
    let mut iter = s2.iter();
    let _y = iter.nth(0);

    let mut s3 = HashSet::new();
    s3.insert(3);
    let mut iter2 = s3.iter();
    let _unwrapped = iter2.nth(0).unwrap();
}
