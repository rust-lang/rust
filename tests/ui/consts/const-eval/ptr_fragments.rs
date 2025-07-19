//! Test that various operations involving pointer fragments work as expected.
//@ run-pass

use std::{mem::{self, MaybeUninit}, ptr};

type Byte = MaybeUninit<u8>;

const unsafe fn memcpy(dst: *mut Byte, src: *const Byte, n: usize) {
    let mut i = 0;
    while i < n {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
}

const _MEMCPY: () = unsafe {
    let ptr = &42;
    let mut ptr2 = ptr::null::<i32>();
    memcpy(&mut ptr2 as *mut _ as *mut _, &ptr as *const _ as *const _, mem::size_of::<&i32>());
    assert!(*ptr2 == 42);
};
const _MEMCPY_OFFSET: () = unsafe {
    // Same as above, but the pointer has a non-zero offset so not all the data bytes are the same.
    let ptr = &(42, 18);
    let ptr = &ptr.1;
    let mut ptr2 = ptr::null::<i32>();
    memcpy(&mut ptr2 as *mut _ as *mut _, &ptr as *const _ as *const _, mem::size_of::<&i32>());
    assert!(*ptr2 == 18);
};

const MEMCPY_RET: MaybeUninit<*const i32> = unsafe {
    let ptr = &42;
    let mut ptr2 = MaybeUninit::new(ptr::null::<i32>());
    memcpy(&mut ptr2 as *mut _ as *mut _, &ptr as *const _ as *const _, mem::size_of::<&i32>());
    // Return in a MaybeUninit so it does not get treated as a scalar.
    ptr2
};

fn main() {
    assert_eq!(unsafe { MEMCPY_RET.assume_init().read() }, 42);
}
