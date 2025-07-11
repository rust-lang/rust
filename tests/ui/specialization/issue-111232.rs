#![feature(min_specialization)]

struct S;

impl From<S> for S { //~ ERROR  conflicting implementations of trait `From<S>` for type `S`
    fn from(s: S) -> S {
        s
    }
}

fn main() {}
