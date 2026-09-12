//@ ignore-compare-mode-next-solver
//@ compile-flags: -Znext-solver
//@ aux-build:opaque-auto-trait-leakage.rs

//! Regression test for https://github.com/rust-lang/rust/issues/134578.
//! Leaking the auto traits of a foreign opaque must not constrain inference
//! variables in the caller, as that would leak the hidden type itself. Here
//! that would constrain `NameMe<T>` to a closure from the auxiliary crate and
//! ICE in typeck. The hidden type may still show up in the diagnostic.

#![feature(type_alias_impl_trait)]
#![allow(unused)]

extern crate opaque_auto_trait_leakage as dep;

use dep::*;

fn require_auto<T: Unpin>(x: T) -> T {
    x
}

type NameMe<T> = impl Sized;

#[define_opaque(NameMe)]
fn leak<T>() -> NameMe<T>
where
    T: Leak<Assoc = NameMe<T>>,
{
    // Proving `impl Sized: Unpin` must not constrain `NameMe<T>`
    // to the foreign closure hidden inside `define`.
    let opaque = require_auto(define::<T>());
    //~^ ERROR type mismatch resolving `<T as Leak>::Assoc == {closure@define<T>::{closure#0}}`
    let closure;
    loop {}
    return closure;
}

fn main() {}
