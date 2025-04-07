//@ known-bug: #136175
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {}

struct A<T>(T)
where
    [(); size_of::<T>()]:;

fn main() {
    let x: A<dyn Trait>;
}
