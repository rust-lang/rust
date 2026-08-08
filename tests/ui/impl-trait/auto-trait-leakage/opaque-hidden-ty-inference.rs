//@ ignore-compare-mode-next-solver
//@ compile-flags: -Znext-solver
//@ aux-build:opaque-auto-trait-leakage.rs

//! Regression test for https://github.com/rust-lang/rust/issues/134578.
//! When reporting a failed auto-trait obligation for an opaque type, diagnostics
//! must not reveal the opaque's hidden type. In this test, the error should refer
//! only to `impl Sized`, without exposing the concrete type from the auxiliary crate.

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
    //~^ ERROR `impl Sized` cannot be unpinned
    let closure;
    loop {}
    return closure;
}

fn main() {}
