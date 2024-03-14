#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(const_mut_refs)]
#![deny(const_eval_mutable_ptr_in_final_value)]
use std::intrinsics;

const BAR: *mut i32 = unsafe { intrinsics::const_allocate(4, 4) as *mut i32 };
//~^ error: mutable pointer in final value of constant
//~| WARNING this was previously accepted by the compiler

fn main() {}
