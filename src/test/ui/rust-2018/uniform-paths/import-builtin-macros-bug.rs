// edition:2018
// aux-build:export-builtin-macros.rs

#![feature(builtin_macro_imports)]

extern crate export_builtin_macros;

mod local {
    pub use concat as my_concat;
}

use export_builtin_macros::*;
use local::*;

fn main() {
    // `concat`s imported from different crates should ideally have the same `DefId`
    // and not conflict with each other, but that's not the case right now.
    my_concat!("x"); //~ ERROR `my_concat` is ambiguous
}
