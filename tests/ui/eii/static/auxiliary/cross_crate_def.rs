//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64;

#[eii1]
pub static EII1_IMPL: u64 = 5;
