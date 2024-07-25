//! This test shows how we fail selection in a way that can influence
//! selection in a code path that succeeds.

//@ revisions: next old
//@[next] compile-flags: -Znext-solver

#![feature(freeze)]

use std::marker::Freeze;

fn is_trait<T: Trait<U>, U: Default>(_: T) -> U {
    Default::default()
}

trait Trait<T> {}
impl<T: Freeze> Trait<u32> for T {}
impl<T> Trait<i32> for T {}
fn foo() -> impl Sized {
    if false { is_trait(foo()) } else { Default::default() }
    //~^ ERROR: type annotations needed
}

fn main() {}
