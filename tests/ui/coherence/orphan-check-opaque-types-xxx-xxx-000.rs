//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

#![feature(type_alias_impl_trait)]

type Opaque0 = impl Sized;
#[define_opaque(Opaque0)] fn define0() -> Opaque0 { 0i32 }

impl foreign::Trait1<(), ()> for Opaque0 {}
//[current]~^ ERROR opaque type `Opaque0` must be used as the argument to some local type
//[next]~^^ ERROR type parameters and opaque types must be used as the argument to some local type

struct Local;

impl foreign::Trait1<Local, ()> for Opaque0 {}
//[current]~^ ERROR opaque type `Opaque0` must be covered by another type when it appears before the first local type
//[next]~^^ ERROR type parameters and opaque types must be covered by another type when it appears before the first local type

type Fundamental<T> = Box<T>; // or any other fundamental type constructor

impl foreign::Trait1<(), Local> for Fundamental<Opaque0> {}
//[current]~^ ERROR opaque type `Opaque0` must be covered by another type when it appears before the first local type
//[next]~^^ ERROR type parameters and opaque types must be covered by another type when it appears before the first local type

type Opaque1 = Fundamental<impl Sized>;
#[define_opaque(Opaque1)] fn define1() -> Opaque1 { Fundamental::new(()) }

// Regression test for <https://github.com/rust-lang/rust/issues/136188>.
impl foreign::Trait1<Local, Local> for Opaque1 {}
//[current]~^ ERROR opaque type `Opaque1::{opaque#0}` must be covered by another type when it appears before the first local type
//[next]~^^ ERROR type parameters and opaque types must be covered by another type when it appears before the first local type

fn main() {}
