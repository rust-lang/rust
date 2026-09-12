// We are std.
#![feature(needs_panic_runtime, no_core)]
#![allow(internal_features)]
#![no_std]
#![no_core]
// Tell rustc to inject panic runtime.
#![needs_panic_runtime]
#![crate_type = "rlib"]

extern crate core;
pub use core::*;
