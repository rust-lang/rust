//! Regression test for ICEs #123959, #125680, #129425 and #136175.
//@ edition:2021

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Trait {}

fn foo<'a, T: 'a>(_: [(); std::mem::offset_of!((T,), 0)]) {}
//~^ ERROR overly complex generic constant
//~| ERROR cycle detected when evaluating type-level constant

struct Inline<T>
//~^ ERROR type parameter `T` is never used
where
    [(); std::mem::offset_of!((T,), 0)]:,
    //~^ ERROR overly complex generic constant
{}

fn main() {
    let dst: Inline<dyn Trait>;
    //~^ ERROR the size for values of type `dyn Trait` cannot be known at compilation time
}
