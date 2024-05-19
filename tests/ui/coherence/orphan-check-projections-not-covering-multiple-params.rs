//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

//@ check-pass
//@ compile-flags: --crate-type=lib
//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

trait Trait<T, U> { type Assoc; }

impl<T, U> Trait<T, U> for () {
    type Assoc = LocalTy;
}

struct LocalTy;

impl<T, U> foreign::Trait0<LocalTy, T, U> for <() as Trait<T, U>>::Assoc {}
//~^ WARNING type parameter `T` must be covered by another type
//~| WARNING this was previously accepted by the compiler
//~| WARNING type parameter `U` must be covered by another type
//~| WARNING this was previously accepted by the compiler


fn main() {}
