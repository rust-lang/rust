//@ run-pass
#![allow(dead_code)]
//@ aux-build:impl-extern-trait-with-associated-type.rs
//! Regression test for https://github.com/rust-lang/rust/issues/20389
//! This test confirms that code implementing a trait with an associated type from an external crate
//! runs.

extern crate impl_extern_trait_with_associated_type;

struct Foo;

impl impl_extern_trait_with_associated_type::T for Foo {
    type C = ();
}

fn main() {}
