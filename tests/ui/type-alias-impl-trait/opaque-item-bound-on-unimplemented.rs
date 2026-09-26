//! Regression test for #162093.
//!
//! When the hidden type of an opaque does not satisfy the opaque's item bounds,
//! the error should name the unsatisfied bound — and let
//! `#[diagnostic::on_unimplemented]` apply — rather than reporting an opaque
//! type mismatch.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

#[diagnostic::on_unimplemented(message = "my custom message")]
trait Marker {}

type Ta = impl Marker;

#[define_opaque(Ta)]
fn construct() -> Ta {
    //~^ ERROR my custom message
    5u32
}

fn main() {}
