//@ run-pass

// This does not reflect a stable guarantee (we guarantee very little for equality of pointers
// around `const`), but it would be good to understand what is happening if these assertions ever
// fail.
use std::ptr::NonNull;
use std::slice::from_raw_parts;

const PTR_U8: *const u8 = NonNull::dangling().as_ptr();
const CONST_U8_REF: &[u8] = unsafe { from_raw_parts(PTR_U8, 0) };
const CONST_U8_PTR: *const u8 = unsafe { from_raw_parts(PTR_U8, 0).as_ptr() };
static STATIC_U8_REF: &[u8] = unsafe { from_raw_parts(PTR_U8, 0) };

const PTR_U16: *const u16 = NonNull::dangling().as_ptr();
const CONST_U16_REF: &[u16] = unsafe { from_raw_parts(PTR_U16, 0) };

const fn const_u8_fn() -> &'static [u8] {
    unsafe { from_raw_parts(PTR_U8, 0) }
}

fn main() {
    let ptr_u8 = unsafe { from_raw_parts(PTR_U8, 0) }.as_ptr();
    let ptr_u16 = unsafe { from_raw_parts(PTR_U16, 0) }.as_ptr();

    assert_eq!(ptr_u8, PTR_U8);
    assert_eq!(ptr_u8, CONST_U8_PTR);
    assert_eq!(ptr_u8, const_u8_fn().as_ptr());
    assert_eq!(ptr_u8, STATIC_U8_REF.as_ptr());
    assert_eq!(ptr_u16, CONST_U16_REF.as_ptr());
    assert_eq!(ptr_u8, CONST_U8_REF.as_ptr());
}
