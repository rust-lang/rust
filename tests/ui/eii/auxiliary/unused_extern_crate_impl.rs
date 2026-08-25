//@ no-prefer-dynamic
//@ aux-build:unused_extern_crate_decl.rs
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

extern crate unused_extern_crate_decl;

#[unused_extern_crate_decl::eii1]
fn impl1(x: u64) {}
