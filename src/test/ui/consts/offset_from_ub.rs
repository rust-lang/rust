#![feature(const_ptr_offset_from)]
#![feature(core_intrinsics)]

use std::intrinsics::ptr_offset_from;

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
    //~| cannot compute offset of pointers into different allocations.
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
    //~| null pointer is not a valid pointer
};

pub const DIFFERENT_INT: isize = { // offset_from with two different integers: like DIFFERENT_ALLOC
    let ptr1 = 8 as *const u8;
    let ptr2 = 16 as *const u8;
    unsafe { ptr_offset_from(ptr2, ptr1) } //~ERROR evaluation of constant value failed
    //~| 0x10 is not a valid pointer
};

fn main() {}
