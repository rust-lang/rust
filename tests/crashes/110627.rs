//@ known-bug: #110627
#![feature(non_lifetime_binders)]

fn take(id: impl for<T> Fn(T) -> T) {}

fn main() {
    take(|x| x)
}
