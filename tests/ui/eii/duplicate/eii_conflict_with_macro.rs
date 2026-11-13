//@ compile-flags: --crate-type rlib
//@ build-pass
#![feature(extern_item_impls)]
#![feature(extern_item_default_impls)]

macro_rules! foo_impl { () => {} }
#[eii]
fn foo_impl() {}
