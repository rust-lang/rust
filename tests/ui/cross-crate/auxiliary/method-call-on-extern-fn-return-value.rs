//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/51798
#![crate_type = "lib"]

pub fn vec() -> Vec<u8> { vec![] }
