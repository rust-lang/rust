//@ no-prefer-dynamic
//@[dylib] compile-flags: --crate-type=dylib -Cprefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64 = 5;
