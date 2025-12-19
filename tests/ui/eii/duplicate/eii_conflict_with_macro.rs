//@ compile-flags: --crate-type rlib
#![feature(extern_item_impls)]

macro_rules! foo_impl {}
#[eii]
fn foo_impl() {}
