//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/18913
//@ no-prefer-dynamic

#![crate_type = "rlib"]
#![crate_name = "foo"]

pub fn foo() -> i32 { 1 }
