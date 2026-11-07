#![feature(non_lifetime_binders)]

//! Regression test for https://github.com/rust-lang/rust/issues/132055.
//! Compiler gave ICE with late-bound type parameter in nested `impl Trait` on
//! associated type.

trait Trait<T: ?Sized> {
    type Assoc<'a> = i32;
    //~^ ERROR associated type defaults are unstable
}

fn produce() -> impl for<T> Trait<(), Assoc = impl Trait<T>> {
    //~^ ERROR cannot capture late-bound type parameter in nested `impl Trait`
    //~| ERROR missing generics for associated type `Trait::Assoc`
    //~| ERROR missing generics for associated type `Trait::Assoc`
    //~| ERROR the trait bound `{integer}: Trait<()>` is not satisfied
    16
}
//~^ ERROR `main` function not found in crate `late_bound_assoc_type`
