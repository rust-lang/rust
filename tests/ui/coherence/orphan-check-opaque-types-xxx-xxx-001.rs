//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

#![feature(impl_trait_in_assoc_type)]

trait Owner { type Opaque; fn define() -> Self::Opaque; }
impl Owner for () { type Opaque = impl Sized; fn define() -> Self::Opaque { 0i32 } }

struct Local;

impl foreign::Trait1<Local, ()> for <() as Owner>::Opaque {}
//[current]~^ ERROR opaque type `<() as Owner>::Opaque` must be covered by another type when it appears before the first local type
//[next]~^^ ERROR type parameters and opaque types must be covered by another type when it appears before the first local type

fn main() {}
