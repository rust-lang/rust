//@ compile-flags: -Znext-solver
//@ check-pass

use std::{iter, slice};

struct Attr;

fn test<'a, T: Iterator<Item = &'a Attr>>() {}

fn main() {
    test::<iter::Filter<slice::Iter<'_, Attr>, fn(&&Attr) -> bool>>();
}
