//! Test that you can use `#![no_core]` and still import std and core manually.
//!
//! The `#![no_core]` attribute disables the automatic core prelude, but you should
//! still be able to explicitly import both `std` and `core` crates and use types
//! like `Option` normally.

//@ run-pass

#![allow(stable_features)]
#![feature(no_core, core)]
#![no_core]

extern crate core;
extern crate std;

use std::option::Option::Some;

fn main() {
    let a = Some("foo");
    a.unwrap();
}
