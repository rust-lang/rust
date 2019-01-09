// no-prefer-dynamic

#![crate_type = "rlib"]

extern crate custom;

use std::sync::atomic::{ATOMIC_USIZE_INIT, Ordering};

use custom::A;

#[global_allocator]
static ALLOCATOR: A = A(ATOMIC_USIZE_INIT);

pub fn get() -> usize {
    ALLOCATOR.0.load(Ordering::SeqCst)
}
