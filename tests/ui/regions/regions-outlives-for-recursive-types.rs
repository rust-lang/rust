//@ check-pass

#![allow(dead_code)]

trait Bound {
    type Assoc: Bound;
}

struct Recurse<'a, T: Bound> {
    first: &'a T,
    value: &'a Recurse<'a, T::Assoc>,
}

fn main() {}
