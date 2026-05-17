//@ aux-build: decl_with_default.rs
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

extern crate decl_with_default as decl;

#[decl::eii1]
pub static EII1_IMPL: u64 = 10;
