// stderr-per-bitwidth
#![feature(ranged_int)]
#![allow(invalid_value)] // make sure we cannot allow away the errors tested here

use std::mem;
use std::ptr::NonNull;
use std::num::{NonZeroU8, NonZeroUsize};
use std::num::Ranged;

const NON_NULL: NonNull<u8> = unsafe { mem::transmute(1usize) };
const NON_NULL_PTR: NonNull<u8> = unsafe { mem::transmute(&1) };

const NULL_PTR: NonNull<u8> = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

const OUT_OF_BOUNDS_PTR: NonNull<u8> = { unsafe {
    let ptr: &[u8; 256] = mem::transmute(&0u8); // &0 gets promoted so it does not dangle
    // Use address-of-element for pointer arithmetic. This could wrap around to null!
    let out_of_bounds_ptr = &ptr[255]; //~ ERROR evaluation of constant value failed
    mem::transmute(out_of_bounds_ptr)
} };

const NULL_U8: NonZeroU8 = unsafe { mem::transmute(0u8) };
//~^ ERROR it is undefined behavior to use this value
const NULL_USIZE: NonZeroUsize = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}
const UNINIT: NonZeroU8 = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR evaluation of constant value failed
//~| uninitialized

// Also test other uses of Ranged

const BAD_RANGE1: Ranged<u32, {10..=30}> = unsafe { Ranged::new_unchecked(42) };
//~^ ERROR it is undefined behavior to use this value

const BAD_RANGE2: Ranged<u32, {30..=10}> = unsafe { Ranged::new_unchecked(20) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
