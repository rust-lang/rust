//! Regression test for https://github.com/rust-lang/rust/issues/162260.
//! An available derive import should take precedence over same-named traits.

//@ run-rustfix
//@ proc-macro: derive-serialize.rs
//@ edition: 2024

#![allow(dead_code)]

extern crate derive_serialize;

mod traits {
    pub trait Serialize {}
}

#[derive(Serialize)]
//~^ ERROR cannot find derive macro `Serialize` in this scope
//~| ERROR cannot find derive macro `Serialize` in this scope
struct S;

fn main() {}
