//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

#![feature(type_alias_impl_trait, impl_trait_in_assoc_type)]

struct Cover<T>(T);

type Opaque = impl Sized;
#[define_opaque(Opaque)] fn define() -> Opaque { 0i32 }

trait Owner { type Opaque; fn define() -> Self::Opaque; }
impl Owner for () { type Opaque = impl Sized; fn define() -> Self::Opaque { 0i32 } }

impl foreign::Trait0<(), (), ()> for Cover<Opaque> {}
impl foreign::Trait1<(), ()> for Cover<<() as Owner>::Opaque> {}

fn main() {}
