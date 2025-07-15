#![feature(min_specialization)]
#![feature(const_trait_impl)]

trait From<T> {
    fn from(t: T) -> Self;
}

impl<T> From<T> for T {
    fn from(t: T) -> T { t }
}

struct S;

impl From<S> for S {
    fn from(s: S) -> S { //~ ERROR `from` specializes an item from a parent `impl`, but that item is not marked `default`
        s
    }
}

fn main() {}
