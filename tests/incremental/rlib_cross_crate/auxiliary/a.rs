//@ no-prefer-dynamic
//@ compile-flags: -Z query-dep-graph

#![crate_type="rlib"]

#[cfg(rpass1)]
pub type X = u32;

#[cfg(rpass2)]
pub type X = i32;

// this version doesn't actually change anything:
#[cfg(rpass3)]
pub type X = i32;

pub type Y = char;
