// This crate is intentionally empty and a re-export of `rustc_driver_impl` to allow the code in
// `rustc_driver_impl` to be compiled in parallel with other crates.

// tidy-alphabetical-start
#![allow(internal_features)]
#![cfg_attr(doc, recursion_limit = "256")] // FIXME(nnethercote): will be removed by #124141
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

pub use rustc_driver_impl::*;
