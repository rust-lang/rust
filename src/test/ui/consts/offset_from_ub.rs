// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

#![feature(const_raw_ptr_deref)]
#![feature(const_ptr_offset_from)]
#![feature(ptr_offset_from)]

#[repr(C)]
struct Struct {
    data: u8,
    field: u8,
}

pub const DIFFERENT_ALLOC: usize = {
    //~^ NOTE
    let uninit = std::mem::MaybeUninit::<Struct>::uninit();
    let base_ptr: *const Struct = &uninit as *const _ as *const Struct;
    let uninit2 = std::mem::MaybeUninit::<Struct>::uninit();
    let field_ptr: *const Struct = &uninit2 as *const _ as *const Struct;
    let offset = unsafe { field_ptr.offset_from(base_ptr) };
    offset as usize
};

pub const NOT_PTR: usize = {
    //~^ NOTE
    unsafe { (42 as *const u8).offset_from(&5u8) as usize }
};

pub const NOT_MULTIPLE_OF_SIZE: isize = {
    //~^ NOTE
    let data = [5u8, 6, 7];
    let base_ptr = data.as_ptr();
    let field_ptr = &data[1] as *const u8 as *const u16;
    unsafe { field_ptr.offset_from(base_ptr as *const u16) }
};

pub const OFFSET_FROM_NULL: isize = {
    //~^ NOTE
    let ptr = 0 as *const u8;
    unsafe { ptr.offset_from(ptr) }
};

pub const DIFFERENT_INT: isize = { // offset_from with two different integers: like DIFFERENT_ALLOC
    //~^ NOTE
    let ptr1 = 8 as *const u8;
    let ptr2 = 16 as *const u8;
    unsafe { ptr2.offset_from(ptr1) }
};

fn main() {}
