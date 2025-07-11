#![feature(min_specialization)]
#![feature(const_trait_impl)]

trait From<T> {
    fn from(t: T) -> Self;
}

impl<T> From<T> for T {
    fn from(t: T) -> T { t }
}

struct S;

impl From<S> for S { //~ ERROR  conflicting implementations of trait `From<S>` for type `S`
    fn from(s: S) -> S {
        s
    }
}

fn main() {}
