//! Tests that a program with no body does not allocate.
//!
//! The initial runtime should not allocate for performance/binary size reasons.
//!
//! -Cprefer-dynamic=no is required as otherwise #[global_allocator] does nothing.
//@ run-pass
//@ compile-flags: -Cprefer-dynamic=no

#[allow(dead_code)]
#[path = "aborting-alloc.rs"]
mod aux;

fn main() {}
