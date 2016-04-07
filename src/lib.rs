#![feature(btree_range, collections, collections_bound, core_intrinsics, rustc_private)]

// From rustc.
extern crate arena;
#[macro_use] extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate syntax;

// From crates.io.
extern crate byteorder;

mod error;
pub mod interpreter;
mod memory;
mod primval;
