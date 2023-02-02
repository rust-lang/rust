// This crate is intentionally empty and a rexport of `rustc_driver_impl` to allow the code in
// `rustc_driver_impl` to be compiled in parallel with other crates.

#![allow(unused_extern_crates)]
extern crate rustc_driver_impl;

pub use rustc_driver_impl::*;
