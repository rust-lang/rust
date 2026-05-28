//! Regression test for #136175

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {}

struct A<T>(T)
where
    [(); size_of::<T>()]:;

fn main() {
    let x: A<dyn Trait>;
    //~^ ERROR the size for values of type `dyn Trait` cannot be known at compilation time
}
