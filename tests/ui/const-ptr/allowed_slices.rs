//@ run-pass
#![feature(
    slice_from_ptr_range,
    const_slice_from_ptr_range,
)]
use std::{
    mem::MaybeUninit,
    ptr,
    slice::{from_ptr_range, from_raw_parts},
};

// Dangling is ok, as long as it's either for ZST reads or for no reads
pub static S0: &[u32] = unsafe { from_raw_parts(dangling(), 0) };
pub static S1: &[()] = unsafe { from_raw_parts(dangling(), 3) };

// References are always valid of reads of a single element (basically `slice::from_ref`)
pub static S2: &[u32] = unsafe { from_raw_parts(&D0, 1) };
pub static S3: &[MaybeUninit<&u32>] = unsafe { from_raw_parts(&D1, 1) };

// Reinterpreting data is fine, as long as layouts match
pub static S4: &[u8] = unsafe { from_raw_parts((&D0) as *const _ as _, 3) };
// This is only valid because D1 has uninitialized bytes, if it was an initialized pointer,
// that would reinterpret pointers as integers which is UB in CTFE.
pub static S5: &[MaybeUninit<u8>] = unsafe { from_raw_parts((&D1) as *const _ as _, 2) };
// Even though u32 and [bool; 4] have different layouts, D0 has a value that
// is valid as [bool; 4], so this is not UB (it's basically a transmute)
pub static S6: &[bool] = unsafe { from_raw_parts((&D0) as *const _ as _, 4) };

// Structs are considered single allocations,
// as long as you don't reinterpret padding as initialized
// data everything is ok.
pub static S7: &[u16] = unsafe {
    let ptr = (&D2 as *const Struct as *const u16).byte_add(4);

    from_raw_parts(ptr, 3)
};
pub static S8: &[MaybeUninit<u16>] = unsafe {
    let ptr = &D2 as *const Struct as *const MaybeUninit<u16>;

    from_raw_parts(ptr, 6)
};

pub static R0: &[u32] = unsafe { from_ptr_range(dangling()..dangling()) };
// from_ptr_range panics on zst
//pub static R1: &[()] = unsafe { from_ptr_range(dangling(), dangling().byte_add(3)) };
pub static R2: &[u32] = unsafe {
    let ptr = &D0 as *const u32;
    from_ptr_range(ptr..ptr.add(1))
};
pub static R3: &[MaybeUninit<&u32>] = unsafe {
    let ptr = &D1 as *const MaybeUninit<&u32>;
    from_ptr_range(ptr..ptr.add(1))
};
pub static R4: &[u8] = unsafe {
    let ptr = &D0 as *const u32 as *const u8;
    from_ptr_range(ptr..ptr.add(3))
};
pub static R5: &[MaybeUninit<u8>] = unsafe {
    let ptr = &D1 as *const MaybeUninit<&u32> as *const MaybeUninit<u8>;
    from_ptr_range(ptr..ptr.add(2))
};
pub static R6: &[bool] = unsafe {
    let ptr = &D0 as *const u32 as *const bool;
    from_ptr_range(ptr..ptr.add(4))
};
pub static R7: &[u16] = unsafe {
    let d2 = &D2;
    let l = &d2.b as *const u32 as *const u16;
    let r = &d2.d as *const u8 as *const u16;

    from_ptr_range(l..r)
};
pub static R8: &[MaybeUninit<u16>] = unsafe {
    let d2 = &D2;
    let l = d2 as *const Struct as *const MaybeUninit<u16>;
    let r = &d2.d as *const u8 as *const MaybeUninit<u16>;

    from_ptr_range(l..r)
};

// Using valid slice is always valid
pub static R9: &[u32] = unsafe { from_ptr_range(R0.as_ptr_range()) };
pub static R10: &[u32] = unsafe { from_ptr_range(R2.as_ptr_range()) };

const D0: u32 = (1 << 16) | 1;
const D1: MaybeUninit<&u32> = MaybeUninit::uninit();
const D2: Struct = Struct { a: 1, b: 2, c: 3, d: 4 };

const fn dangling<T>() -> *const T {
    ptr::NonNull::dangling().as_ptr() as _
}

#[repr(C)]
struct Struct {
    a: u8,
    // _pad: [MaybeUninit<u8>; 3]
    b: u32,
    c: u16,
    d: u8,
    // _pad: [MaybeUninit<u8>; 1]
}

fn main() {}
