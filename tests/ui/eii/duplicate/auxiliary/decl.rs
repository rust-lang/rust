//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
fn decl1(x: u64);
