// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

#![allow(invalid_value)] // make sure we cannot allow away the errors tested here
#![feature(rustc_attrs, ptr_metadata)]

use std::mem;
use std::ptr::NonNull;
use std::num::NonZero;

const NON_NULL: NonNull<u8> = unsafe { mem::transmute(1usize) };
const NON_NULL_PTR: NonNull<u8> = unsafe { mem::transmute(&1) };

const NULL_PTR: NonNull<u8> = unsafe { mem::transmute(0usize) };
//~^ ERROR invalid value

const OUT_OF_BOUNDS_PTR: NonNull<u8> = { unsafe {
    let ptr: &[u8; 256] = mem::transmute(&0u8); // &0 gets promoted so it does not dangle
    // Use address-of-element for pointer arithmetic. This could wrap around to null!
    let out_of_bounds_ptr = &ptr[255]; //~ ERROR in-bounds pointer arithmetic failed
    mem::transmute(out_of_bounds_ptr)
} };

const NULL_U8: NonZero<u8> = unsafe { mem::transmute(0u8) };
//~^ ERROR invalid value
const NULL_USIZE: NonZero<usize> = unsafe { mem::transmute(0usize) };
//~^ ERROR invalid value

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}
const UNINIT: NonZero<u8> = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR uninitialized

// Also test other uses of rustc_layout_scalar_valid_range_start

#[rustc_layout_scalar_valid_range_start(10)]
#[rustc_layout_scalar_valid_range_end(30)]
struct RestrictedRange1(u32);
const BAD_RANGE1: RestrictedRange1 = unsafe { RestrictedRange1(42) };
//~^ ERROR invalid value

#[rustc_layout_scalar_valid_range_start(30)]
#[rustc_layout_scalar_valid_range_end(10)]
struct RestrictedRange2(u32);
const BAD_RANGE2: RestrictedRange2 = unsafe { RestrictedRange2(20) };
//~^ ERROR invalid value

const NULL_FAT_PTR: NonNull<dyn Send> = unsafe {
//~^ ERROR invalid value
    let x: &dyn Send = &42;
    let meta = std::ptr::metadata(x);
    mem::transmute((0_usize, meta))
};

static S: u32 = 0; // just a static to construct a maybe-null pointer off of
const MAYBE_NULL_PTR: NonNull<()> = unsafe { mem::transmute((&raw const S).wrapping_add(4)) };
//~^ ERROR invalid value

fn main() {}
