#![feature(btree_range, collections_bound, rustc_private)]

// From rustc.
extern crate arena;
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_mir;
extern crate syntax;

// From crates.io.
extern crate byteorder;

mod error;
pub mod interpreter;
mod memory;
mod primval;
