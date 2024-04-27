#![feature(const_ptr_sub_ptr)]
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
    let offset = unsafe { ptr_offset_from(field_ptr, base_ptr) }; //~ERROR evaluation of constant value failed
    //~| pointers into different allocations
    offset as usize
};

pub const NOT_PTR: usize = {
    unsafe { (42 as *const u8).offset_from(&5u8) as usize }
};

pub const NOT_MULTIPLE_OF_SIZE: isize = {
    let data = [5u8, 6, 7];
    let base_ptr = data.as_ptr();
    let field_ptr = &data[1] as *const u8 as *const u16;
    unsafe { ptr_offset_from(field_ptr, base_ptr as *const u16) } //~ERROR evaluation of constant value failed
    //~| 1_isize cannot be divided by 2_isize without remainder
};

pub const OFFSET_FROM_NULL: isize = {
    let ptr = 0 as *const u8;
    unsafe { ptr_offset_from(ptr, ptr) } //~ERROR evaluation of constant value failed
    //~| null pointer is a dangling pointer
};

pub const DIFFERENT_INT: isize = { // offset_from with two different integers: like DIFFERENT_ALLOC
    let ptr1 = 8 as *const u8;
    let ptr2 = 16 as *const u8;
    unsafe { ptr_offset_from(ptr2, ptr1) } //~ERROR evaluation of constant value failed
    //~| 0x8[noalloc] is a dangling pointer
};

const OUT_OF_BOUNDS_1: isize = {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = (start_ptr).wrapping_add(length);
    // First ptr is out of bounds
    unsafe { ptr_offset_from(end_ptr, start_ptr) } //~ERROR evaluation of constant value failed
    //~| pointer to 10 bytes starting at offset 0 is out-of-bounds
};

const OUT_OF_BOUNDS_2: isize = {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = (start_ptr).wrapping_add(length);
    // Second ptr is out of bounds
    unsafe { ptr_offset_from(start_ptr, end_ptr) } //~ERROR evaluation of constant value failed
    //~| pointer to 10 bytes starting at offset 0 is out-of-bounds
};

const OUT_OF_BOUNDS_SAME: isize = {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = (start_ptr).wrapping_add(length);
    unsafe { ptr_offset_from(end_ptr, end_ptr) } //~ERROR evaluation of constant value failed
    //~| pointer at offset 10 is out-of-bounds
};

pub const DIFFERENT_ALLOC_UNSIGNED: usize = {
    let uninit = std::mem::MaybeUninit::<Struct>::uninit();
    let base_ptr: *const Struct = &uninit as *const _ as *const Struct;
    let uninit2 = std::mem::MaybeUninit::<Struct>::uninit();
    let field_ptr: *const Struct = &uninit2 as *const _ as *const Struct;
    unsafe { ptr_offset_from_unsigned(field_ptr, base_ptr) } //~ERROR evaluation of constant value failed
    //~| pointers into different allocations
};

pub const TOO_FAR_APART1: isize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_add(isize::MAX as usize + 42);
    unsafe { ptr_offset_from(ptr2, ptr1) } //~ERROR evaluation of constant value failed
    //~| too far ahead
};
pub const TOO_FAR_APART2: isize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_add(isize::MAX as usize + 42);
    unsafe { ptr_offset_from(ptr1, ptr2) } //~ERROR evaluation of constant value failed
    //~| too far before
};

const WRONG_ORDER_UNSIGNED: usize = {
    let a = ['a', 'b', 'c'];
    let p = a.as_ptr();
    unsafe { ptr_offset_from_unsigned(p, p.add(2) ) } //~ERROR evaluation of constant value failed
    //~| first pointer has smaller offset than second: 0 < 8
};
pub const TOO_FAR_APART_UNSIGNED: usize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_add(isize::MAX as usize + 42);
    // This would fit into a `usize` but we still don't allow it.
    unsafe { ptr_offset_from_unsigned(ptr2, ptr1) } //~ERROR evaluation of constant value failed
    //~| too far ahead
};

// These do NOT complain that pointers are too far apart; they pass that check (to then fail the
// next one).
pub const OFFSET_VERY_FAR1: isize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_offset(isize::MAX);
    unsafe { ptr2.offset_from(ptr1) }
    //~^ inside
};
pub const OFFSET_VERY_FAR2: isize = {
    let ptr1 = ptr::null::<u8>();
    let ptr2 = ptr1.wrapping_offset(isize::MAX);
    unsafe { ptr1.offset_from(ptr2.wrapping_offset(1)) }
    //~^ inside
};

fn main() {}
