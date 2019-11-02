// ignore-x86 FIXME: missing sysroot spans (#53081)

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

pub const NOT_MULTIPLE_OF_SIZE: usize = {
    //~^ NOTE
    let data = [5u8, 6, 7];
    let base_ptr = data.as_ptr();
    let field_ptr = &data[1] as *const u8 as *const u16;
    let offset = unsafe { field_ptr.offset_from(base_ptr as *const u16) };
    offset as usize
};

fn main() {}
