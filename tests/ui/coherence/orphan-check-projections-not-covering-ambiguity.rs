// This test demonstrates a limitation of the trait solver.
// Basically, one might think that `T` was covered by the projection since the
// latter appears to normalize to a local type. However, since we instantiate the
// constituent types of the self type of impls with fresh infer vars and try to
// normalize them during orphan checking, we wind up trying to normalize a
// projection whose self type is an infer var which unconditionally fails due to
// ambiguity.

//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

//@ check-pass
//@ compile-flags: --crate-type=lib
//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

trait Project { type Output; }

impl<T> Project for T {
    type Output = Local;
}

struct Local;

impl<T> foreign::Trait1<Local, T> for <T as Project>::Output {}
//~^ WARNING type parameter `T` must be covered by another type
//~| WARNING this was previously accepted by the compiler

fn main() {}
