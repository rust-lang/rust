// stderr-per-bitwidth
#![feature(rustc_attrs)]
#![allow(const_err, invalid_value)] // make sure we cannot allow away the errors tested here

use std::mem;
use std::ptr::NonNull;
use std::num::{NonZeroU8, NonZeroUsize};

const NON_NULL: NonNull<u8> = unsafe { mem::transmute(1usize) };
const NON_NULL_PTR: NonNull<u8> = unsafe { mem::transmute(&1) };

const NULL_PTR: NonNull<u8> = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

#[deny(const_err)] // this triggers a `const_err` so validation does not even happen
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

// Also test other uses of rustc_layout_scalar_valid_range_start

#[rustc_layout_scalar_valid_range_start(10)]
#[rustc_layout_scalar_valid_range_end(30)]
struct RestrictedRange1(u32);
const BAD_RANGE1: RestrictedRange1 = unsafe { RestrictedRange1(42) };
//~^ ERROR it is undefined behavior to use this value

#[rustc_layout_scalar_valid_range_start(30)]
#[rustc_layout_scalar_valid_range_end(10)]
struct RestrictedRange2(u32);
const BAD_RANGE2: RestrictedRange2 = unsafe { RestrictedRange2(20) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
