// This crate is intentionally empty and a re-export of `rustc_driver_impl` to allow the code in
// `rustc_driver_impl` to be compiled in parallel with other crates.

#![cfg_attr(not(bootstrap), allow(internal_features))]
#![cfg_attr(not(bootstrap), feature(rustdoc_internals))]
#![cfg_attr(not(bootstrap), doc(rust_logo))]

pub use rustc_driver_impl::*;
