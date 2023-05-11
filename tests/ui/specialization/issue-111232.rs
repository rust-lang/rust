#![feature(min_specialization)]

struct S;

impl From<S> for S {
    fn from(s: S) -> S { //~ ERROR `from` specializes an item from a parent `impl`, but that item is not marked `default`
        s
    }
}

fn main() {}
