#![crate_name = "a"]
#![crate_type = "dylib"]

#[cfg(x)]
pub fn foo(x: u32) {}

#[cfg(y)]
pub fn foo(x: i32) {}
