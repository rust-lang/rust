#![feature(no_core)]
#![no_core]

// See rustc-std-workspace-core for why this crate is needed.

// Rename the crate to avoid conflicting with the alloc module in alloc.
extern crate alloc as foo;

pub use foo::*;
