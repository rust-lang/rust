//@ normalize-stderr: "\d+ bytes" -> "$$BYTES bytes"
//@ dont-require-annotations: NOTE

#![feature(core_intrinsics)]

use std::intrinsics::{ptr_offset_from, ptr_offset_from_unsigned};
use std::ptr;

#[repr(C)]
struct Struct {
    data: u8,
    field: u8,
}

pub const DIFFERENT_ALLOC: usize = {
    let uninit = std::mem::MaybeUninit::<Struct>::uninit();
    let base_ptr: *const Struct = &uninit as *const _ as *const Struct;
    let uninit2 = std::mem::MaybeUninit::<Struct>::uninit();
    let field_ptr: *const Struct = &uninit2 as *const _ as *const Struct;
    let offset = unsafe { ptr_offset_from(field_ptr, base_ptr) };
    //~^ ERROR not both derived from the same allocation
    offset as usize
};

pub const NOT_PTR: usize = {
    unsafe { (42 as *const u8).offset_from(&5u8) as usize }
    //~^ ERROR not both derived from the same allocation
};

pub const NOT_MULTIPLE_OF_SIZE: isize = {
    let data = [5u8, 6, 7];
    let base_ptr = data.as_ptr();
    let field_ptr = &data[1] as *const u8 as *const u16;
    unsafe { ptr_offset_from(field_ptr, base_ptr as *const u16) }
    //~^ ERROR 1_isize cannot be divided by 2_isize without remainder
};

pub const DIFFERENT_INT: isize = {
    // offset_from with two different integers: like DIFFERENT_ALLOC
    let ptr1 = 8 as *const u8;
    let ptr2 = 16 as *const u8;
    unsafe { ptr_offset_from(ptr2, ptr1) }
    //~^ ERROR not both derived from the same allocation
};

const OUT_OF_BOUNDS_1: isize = {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = (start_ptr).wrapping_add(length);
    // First ptr is out of bounds
    unsafe { ptr_offset_from(end_ptr, start_ptr) }
    //~^ ERROR the memory range between them is not in-bounds of an allocation
};

const OUT_OF_BOUNDS_2: isize = {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = (start_ptr).wrapping_add(length);
    // Second ptr is out of bounds
    unsafe { ptr_offset_from(start_ptr, end_ptr) }
    //~^ ERROR the memory range between them is not in-bounds of an allocation
};

pub const DIFFERENT_ALLOC_UNSIGNED: usize = {
    let uninit = std::mem::MaybeUninit::<Struct>::uninit();
    let base_ptr: *const Struct = &uninit as *const _ as *const Struct;
    let uninit2 = std::mem::MaybeUninit::<Struct>::uninit();
    let field_ptr: *const Struct = &uninit2 as *const _ as *const Struct;
    unsafe { ptr_offset_from_unsigned(field_ptr, base_ptr) }
    //~^ ERROR not both derived from the same allocation
};

pub const TOO_FAR_APART1: isize = {
    let ptr1 = &0u8 as *const u8;
    let ptr2 = ptr1.wrapping_add(isize::MAX as usize + 42);
    unsafe { ptr_offset_from(ptr2, ptr1) }
    //~^ ERROR too far ahead
};
pub const TOO_FAR_APART2: isize = {
    let ptr1 = &0u8 as *const u8;
    let ptr2 = ptr1.wrapping_add(isize::MAX as usize + 42);
    unsafe { ptr_offset_from(ptr1, ptr2) }
    //~^ ERROR too far before
};
pub const TOO_FAR_APART3: isize = {
    let ptr1 = &0u8 as *const u8;
    let ptr2 = ptr1.wrapping_offset(isize::MIN);
    // The result of this would be `isize::MIN`, which *does* fit in an `isize`, but its
    // absolute value does not. (Also anyway there cannot be an allocation of that size.)
    unsafe { ptr_offset_from(ptr1, ptr2) }
    //~^ ERROR too far before
};

const WRONG_ORDER_UNSIGNED: usize = {
    let a = ['a', 'b', 'c'];
    let p = a.as_ptr();
    unsafe { ptr_offset_from_unsigned(p, p.add(2)) }
    //~^ ERROR first pointer has smaller offset than second: 0 < 8
};
pub const TOO_FAR_APART_UNSIGNED: usize = {
    let ptr1 = &0u8 as *const u8;
    let ptr2 = ptr1.wrapping_add(isize::MAX as usize + 42);
    // This would fit into a `usize` but we still don't allow it.
    unsafe { ptr_offset_from_unsigned(ptr2, ptr1) } //~ERROR too far ahead
};

// These do NOT complain that pointers are too far apart; they pass that check (to then fail the
// next one).
pub const OFFSET_VERY_FAR1: isize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_offset(isize::MAX);
    unsafe { ptr2.offset_from(ptr1) }
    //~^ ERROR called on two different pointers that are not both derived from the same allocation
};
pub const OFFSET_VERY_FAR2: isize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_offset(isize::MAX);
    unsafe { ptr1.offset_from(ptr2.wrapping_offset(1)) }
    //~^ ERROR ptr_offset_from` called when first pointer is too far before second
};

// If the pointers are the same, OOB/null/UAF is fine.
pub const OFFSET_FROM_NULL_SAME: isize = {
    let ptr = 0 as *const u8;
    unsafe { ptr_offset_from(ptr, ptr) }
};
const OUT_OF_BOUNDS_SAME: isize = {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = (start_ptr).wrapping_add(length);
    unsafe { ptr_offset_from(end_ptr, end_ptr) }
};
const UAF_SAME: isize = {
    let uaf_ptr = {
        let x = 0;
        &x as *const i32
    };
    unsafe { ptr_offset_from(uaf_ptr, uaf_ptr) }
};

fn main() {}
