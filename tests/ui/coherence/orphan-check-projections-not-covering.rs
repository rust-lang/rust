// Regression test for issue #99554.
// Projections might not cover type parameters.

//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

//@ compile-flags: --crate-type=lib
//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

trait Identity {
    type Output;
}

impl<T> Identity for T {
    type Output = T;
}

struct Local;

impl<T> foreign::Trait0<Local, T, ()> for <T as Identity>::Output {}
//~^ ERROR type parameter `T` must be covered by another type

impl<T> foreign::Trait0<<T as Identity>::Output, Local, T> for Option<T> {}
//~^ ERROR type parameter `T` must be covered by another type

pub trait Deferred {
    type Output;
}

// A downstream user could implement
//
//     impl<T> Deferred for Type<T> { type Output = T; }
//     struct Type<T>(T);
//
impl<T: Deferred> foreign::Trait1<Local, T> for <T as Deferred>::Output {}
//~^ ERROR type parameter `T` must be covered by another type
