// Version of `tests/ui/consts/const-eval/heap/alloc_intrinsic_untyped.rs` without the flag that
// suppresses the ICE.
//@ known-bug: #129233
#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(const_mut_refs)]
use std::intrinsics;

const BAR: *mut i32 = unsafe { intrinsics::const_allocate(4, 4) as *mut i32 };

fn main() {}
