//@ compile-flags:--crate-name=test
//@ aux-build:coherence_lib.rs

// Ensure that `Box` in particular isn't fundamental over
// the allocator parameter (but is over T).

#![feature(allocator_api)]

extern crate coherence_lib as lib;

use lib::*;
use std::alloc::{Allocator, AllocError, Layout};
use std::ptr::NonNull;

struct Local;

unsafe impl Allocator for Local {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

impl Remote for Box<str, Local> {}
  //~^ ERROR: only traits defined in the current crate can be implemented for types defined outside of the crate [E0117]

fn main() {}
