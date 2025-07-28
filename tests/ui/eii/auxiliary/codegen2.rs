//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(eii)]

#[eii(eii1)]
pub fn decl1(x: u64);
