//@ edition: 2024
//@ compile-flags: -Znext-solver -Zdisable-fast-paths
//@ check-pass

// Test whether we mark the rigid alias in hir typeck non-rigid in const eval
// where we reveal all hidden types with `PostAnalysis` typing mode.
//
// This shouldn't compile as we shouldn't do const eval in non-empty param env
// in the future.

#![feature(type_alias_impl_trait)]
type Tait<T> = impl Sized;
#[define_opaque(Tait)]
fn foo<T>(x: T) -> Tait<T> {
    x
}

trait Trait {
    type Assoc;
}

fn bar<T>() -> [u8; 4]
where
    T: Trait,
    Tait<T>: Trait<Assoc = u32>,
{
    [0; std::mem::size_of::<<T as Trait>::Assoc>()]
    //~^ WARN: cannot use constants which depend on generic parameters in types
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
