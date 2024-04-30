//! Tests that a program with no body does not allocate.
//!
//! The initial runtime should not allocate for performance/binary size reasons.
//!
//! -Cprefer-dynamic=no is required as otherwise #[global_allocator] does nothing.
//! We only test linux-gnu as other targets currently need allocation for thread dtors.
//@ run-pass
//@ compile-flags: -Cprefer-dynamic=no -Cdebuginfo=full
//@ only-linux
//@ only-gnu

#[allow(dead_code)]
#[path = "aborting-alloc.rs"]
mod aux;

fn main() {}
