#![feature(
    btree_range,
    collections,
    collections_bound,
    core_intrinsics,
    filling_drop,
    rustc_private,
)]

// From rustc.
#[macro_use] extern crate rustc;
extern crate rustc_mir;
extern crate syntax;

// From crates.io.
extern crate byteorder;

mod error;
pub mod interpreter;
mod memory;
mod primval;
