// Opaque types never cover type parameters.

//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

#![feature(type_alias_impl_trait)]

type Identity<T> = impl Sized;

#[define_opaque(Identity)]
fn define_identity<T>(x: T) -> Identity<T> {
    x
}

impl<T> foreign::Trait0<Local, T, ()> for Identity<T> {}
//~^ ERROR type parameter `T` must be covered by another type

type Opaque<T> = impl Sized;

#[define_opaque(Opaque)]
fn define_local<T>() -> Opaque<T> {
    Local
}

impl<T> foreign::Trait1<Local, T> for Opaque<T> {}
//~^ ERROR type parameter `T` must be covered by another type

struct Local;

fn main() {}
