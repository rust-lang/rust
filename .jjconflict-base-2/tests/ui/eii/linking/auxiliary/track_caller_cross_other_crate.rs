//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[track_caller]
#[eii(tcross)]
pub fn tcross(x: u64);
