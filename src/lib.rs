#![feature(btree_range, collections_bound, rustc_private)]

extern crate byteorder;
extern crate rustc;
extern crate rustc_mir;
extern crate syntax;

pub mod interpreter;
mod memory;
mod primval;
