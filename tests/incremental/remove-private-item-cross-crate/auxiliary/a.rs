#![allow(warnings)]
#![crate_name = "a"]
#![crate_type = "rlib"]

pub fn foo(b: u8) -> u32 { b as u32 }

#[cfg(rpass1)]
fn bar() { }
