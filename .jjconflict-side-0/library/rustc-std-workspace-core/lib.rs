#![feature(no_core)]
#![no_core]

pub use core::*;

// Crate must be brought into scope so it appears in the crate graph for anything that
// depends on `rustc-std-workspace-core`.
use compiler_builtins as _;
