//@ run-pass

struct Struct {
    field: (),
}

#[repr(C)]
struct Struct2 {
    data: u8,
    field: u8,
}

pub const OFFSET: usize = {
    let uninit = std::mem::MaybeUninit::<Struct>::uninit();
    let base_ptr: *const Struct = &uninit as *const _ as *const Struct;
    // The following statement is UB (taking the address of an uninitialized field).
    // Const eval doesn't detect this right now, but it may stop compiling at some point
    // in the future.
    let field_ptr = unsafe { &(*base_ptr).field as *const () as *const u8 };
    let offset = unsafe { field_ptr.offset_from(base_ptr as *const u8) };
    offset as usize
};

pub const OFFSET_2: usize = {
    let uninit = std::mem::MaybeUninit::<Struct2>::uninit();
    let base_ptr: *const Struct2 = &uninit as *const _ as *const Struct2;
    let field_ptr = unsafe { &(*base_ptr).field as *const u8 };
    let offset = unsafe { field_ptr.offset_from(base_ptr as *const u8) };
    offset as usize
};

pub const OVERFLOW: isize = {
    let uninit = std::mem::MaybeUninit::<Struct2>::uninit();
    let base_ptr: *const Struct2 = &uninit as *const _ as *const Struct2;
    let field_ptr = unsafe { &(*base_ptr).field as *const u8 };
    unsafe { (base_ptr as *const u8).offset_from(field_ptr) }
};

pub const OFFSET_EQUAL_INTS: isize = {
    let ptr = 1 as *const u8;
    unsafe { ptr.offset_from(ptr) }
};

pub const OFFSET_UNSIGNED: usize = {
    let a = ['a', 'b', 'c'];
    let ptr = a.as_ptr();
    unsafe { ptr.add(2).offset_from_unsigned(ptr) }
};

fn main() {
    assert_eq!(OFFSET, 0);
    assert_eq!(OFFSET_2, 1);
    assert_eq!(OVERFLOW, -1);
    assert_eq!(OFFSET_EQUAL_INTS, 0);
    assert_eq!(OFFSET_UNSIGNED, 2);
}
